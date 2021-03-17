import logging
import time
import os

import torch
import torch.nn as nn
import torchsummary
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment
from joblib import dump, load
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import plotly.offline as plotlyoff
import plotly.graph_objs as plotlygo

from lib.modules.unet import UNet

from lib.modules.discriminative_loss import DiscriminativeLoss
# from lib.modules.discriminative_loss_withforloop import DiscriminativeLoss


logger = logging.getLogger(__name__)


class PixelwiseModel:
    """Defines training, evaluation, ... for a **pixel-wise model**"""
    def __init__(self,
                 backbone_model_name,
                 backbone_model_params,
                 padding,
                 load_model_path=None):

        if backbone_model_name == 'unet':
            unet = UNet(*backbone_model_params, padding=padding)
        else:
            raise NotImplementedError()
        self.model = unet
        self.best_loss = np.inf
        self.step = 0
        self.load_model_path = load_model_path
        if load_model_path:
            assert os.path.exists(load_model_path), f'load_model_path:[{load_model_path}], does not exist.'
            self.step, self.best_loss = self.load_model(load_model_path)
        self.model.cuda()

    def save_model(self, step, optimizer, lr_scheduler, loss, filename):
        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': loss
            }, filename)
        logger.info(f'Saved model to:[{filename}]. step:[{step}]. loss:[{loss}]')

    def load_model(self, filename, optimizer=None, lr_scheduler=None):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if lr_scheduler:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        step = checkpoint['step']
        loss = checkpoint['loss']
        logger.info(f'Loaded model from:[{filename}]. step:[{step}]. loss:[{loss}]')
        return step, loss

    def print_model_summary(self, sample_input):
        # TODO: summary only prints to stdout, does not return str for logger
        input_size = sample_input.size()[1:]
        print('Backbone Summary:')
        torchsummary.summary(self.model, input_size)
        # Do not need this, since torchsummary does not count weight sharing -> wrong trainable param information, etc.
        # print('Siamese  Summary:')
        # torchsummary.summary(self.model, [input_size, input_size])

    @staticmethod
    def _define_criterion():
        # TODO: currently only loss with default values
        return DiscriminativeLoss()

    def _define_optimizer(self, learning_rate, weight_decay, lr_drop_factor, lr_drop_patience, load_state=True):
        parameters = self.model.parameters()
        optimizer = torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay
            )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=lr_drop_factor, patience=lr_drop_patience, verbose=True
            )
        if load_state:
            if self.load_model_path:
                checkpoint = torch.load(self.load_model_path)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        return optimizer, lr_scheduler

    def _train_one_iteration(self, input, criterion, optimizer):
        raw = input['raw'].float().cuda()
        label = input['label'].float().cuda()

        optimizer.zero_grad()
        out = self.model(raw)

        loss, l_var, l_dist, l_reg = criterion(out, label)
        assert not torch.isnan(loss), 'Why is loss NaN sometimes?'
        if loss.item() > 100:
            logger.debug('SKIPPING TRAINING - NOT TRAINING FOR THIS STEP BECAUSE LOSS TOO LARGE')
            return loss, l_var, l_dist, l_reg

        loss.backward()
        optimizer.step()

        return loss, l_var, l_dist, l_reg

    def fit(self, train_loader, n_step, learning_rate, weight_decay, lr_drop_factor, lr_drop_patience,
            model_ckpt_every_n_step, model_save_path, running_loss_interval, burn_in_step, tb_writer=None):
        criterion = self._define_criterion()
        optimizer, lr_scheduler = self._define_optimizer(learning_rate, weight_decay, lr_drop_factor, lr_drop_patience)

        logger.info(f'Start training from step:{self.step} to step:{n_step}')

        train_iter = iter(train_loader)

        if tb_writer:
            tmp_input = next(train_iter)
            tb_writer.add_graph(self.model, tmp_input['raw'].float().cuda())

        self.model.train()

        running_loss_timer = time.time()
        running_loss = np.zeros(running_loss_interval)
        for step in range(self.step+1, n_step+1):
            inputs = next(train_iter)
            loss, l_var, l_dist, l_reg = self._train_one_iteration(inputs, criterion, optimizer)

            running_loss[:-1] = running_loss[1:]; running_loss[-1] = loss.item()

            tb_writer.add_scalar('Loss/train', loss.item(), step)
            tb_writer.add_scalar('Loss/train_lvar', l_var.item(), step)
            tb_writer.add_scalar('Loss/train_ldist', l_dist.item(), step)
            tb_writer.add_scalar('Loss/train_lreg', l_reg.item(), step)
            tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)

            if step % model_ckpt_every_n_step == 0:
                self.save_model(step, optimizer, lr_scheduler, running_loss.mean(),
                                os.path.join(model_save_path, f'model-step={step}.pth'))

            if step % running_loss_interval == 0:
                avg_running_loss = running_loss.mean()
                if avg_running_loss < self.best_loss:
                    self.best_loss = avg_running_loss
                    if step > burn_in_step:
                        # Saving best model
                        existing_best_model = [os.path.join(model_save_path, f) for f in os.listdir(model_save_path) if
                                               f.startswith('bestmodel')]
                        if existing_best_model:
                            os.remove(existing_best_model[0])
                        self.save_model(step, optimizer, lr_scheduler, avg_running_loss,
                                        os.path.join(model_save_path, f'bestmodel-step={step}-running_loss={avg_running_loss:.4f}.pth'))

                # preferably do it every epoch, but since no epoch meaning, use running_loss_interval instead
                if tb_writer:
                    for param_name, param in self.model.named_parameters():
                        tb_writer.add_histogram(param_name, param.data, step)

                # TODO: best case scenario should be a validation loss or metric of some sort, but running_loss for now
                lr_scheduler.step(avg_running_loss)

                logger.info(f'steps [{step}/{n_step}] - running_loss [{avg_running_loss:.5f}] - time ['
                            f'{time.time()-running_loss_timer:.2f}s]')
                running_loss_timer = time.time()

    @staticmethod
    def get_avg_embedding_over_mask(batched_embedding , batched_mask):
        output_list = []
        bs, n_features, xsize, ysize, zsize = batched_embedding.size()
        batched_embedding = batched_embedding.contiguous().view(bs, n_features, xsize*ysize*zsize)
        batched_mask = batched_mask.contiguous().view(bs, xsize*ysize*zsize)
        for i in range(bs):
            output_list.append(
                batched_embedding[i, :, :].sum(1) / batched_mask[i, :].sum()
                )
        return output_list

    def compute_cluster_centers(self, n_cluster, data_loader, cluster_save_file=None, num_workers=1, tb_writer=None,
                                save_embedding_image_file_path=None):
        all_avg_embeddings_per_seghyp = []
        all_gt_labels = []
        self.model.eval()
        for n_worm, worm_dataset in enumerate(data_loader):
            logger.info(f'worm: {n_worm+1}/{len(data_loader)} - n_seghyp={len(worm_dataset)}')
            worm_loader = torch.utils.data.DataLoader(worm_dataset, shuffle=False, num_workers=num_workers)
            for n_seghyp, seghyp in enumerate(worm_loader):
                # TODO: added last secondinsh, check if keep or remove
                if torch.any(seghyp['mask']==-1):
                    logger.debug('Skipped successfully by checking -1 in mask - n-seghyp was {}'.format(n_seghyp))
                    continue
                with torch.no_grad():
                    raw = seghyp['raw'].float().cuda()
                    mask = seghyp['mask'].float().cuda()
                    gt_label_id = seghyp['gt_label_id'].cuda()

                    batched_embedding = self.model(raw)
                    batched_avg_embedding = self.get_avg_embedding_over_mask(batched_embedding, mask)
                    all_avg_embeddings_per_seghyp.extend(
                        [avg_embd.cpu().numpy() for avg_embd in batched_avg_embedding])
                    all_gt_labels.extend(list(gt_label_id.cpu().numpy()))

        all_avg_embeddings_per_seghyp = np.vstack(all_avg_embeddings_per_seghyp)
        all_gt_labels = np.squeeze(np.vstack(all_gt_labels))

        kmeans = KMeans(n_clusters=n_cluster).fit(all_avg_embeddings_per_seghyp)
        logger.info(f'Save scipy.cluster.KMeans model at: [{cluster_save_file}]')
        os.makedirs(os.path.dirname(cluster_save_file), exist_ok=True)
        dump(kmeans, cluster_save_file)

        # plot t-sne embeddings. plot seaborn to tb_write and plotly (mybe even 3d) to saved image
        if tb_writer or save_embedding_image_file_path:
            r_clist = np.random.rand(559, 3)
            r_clist[0] = [0, 0, 0]
            cluster_centers = kmeans.cluster_centers_

        if tb_writer:
            tsne = TSNE(n_components=2)
            tsne_results = tsne.fit_transform(all_avg_embeddings_per_seghyp)
            fig = plt.figure()

            for i in range(559):
                selected_gt_labels = np.where(all_gt_labels==i)
                plt.scatter(x=tsne_results[selected_gt_labels,0], y=tsne_results[selected_gt_labels,1],
                            c=np.expand_dims(r_clist[i], axis=0),
                            s=40, alpha=0.5)
            tsne_cluster_centers = tsne.fit_transform(cluster_centers)
            plt.scatter(x=tsne_cluster_centers[:,0], y=tsne_cluster_centers[:,1], s=80, alpha=0.4)
            tb_writer.add_figure('avg_seghyp_embdd-TSNE', fig, global_step=self.step)
        if save_embedding_image_file_path:
            r_clist *= 255
            os.makedirs(os.path.dirname(save_embedding_image_file_path), exist_ok=True)
            tsne = TSNE(n_components=3)
            tsne_results = tsne.fit_transform(all_avg_embeddings_per_seghyp)
            tsne_cluster_centers = tsne.fit_transform(cluster_centers)
            trace_data = []
            trace_data.append(plotlygo.Scatter3d(
                x=tsne_results[:, 0],
                y=tsne_results[:, 1],
                z=tsne_results[:, 2],
                mode='markers',
                showlegend=False,
                hovertext=all_gt_labels.astype(str),
                marker=dict(
                    size=5,
                    color=[f'rgb({r_clist[i][0]},{r_clist[i][1]},{r_clist[i][2]})' for i in all_gt_labels],
                    # color=all_gt_labels.astype(str),
                    # showscale=False,
                    # line=dict(
                    #     width=2,
                    #     color='rgb(255, 255, 255)'
                    #     ),
                    opacity=0.8
                    )
                ))
            trace_data.append(
                plotlygo.Scatter3d(
                    x=tsne_cluster_centers[:,0],
                    y=tsne_cluster_centers[:,1],
                    z=tsne_cluster_centers[:,2],
                    mode='markers',
                    showlegend=False,
                    marker=dict(
                        size=10,
                        color='black',
                        opacity=0.4
                        )
                    )
                )
            data = trace_data
            layout = dict(title=f'avg_seghyp_embd - step={self.step}',
                          hovermode='closest',
                          yaxis=dict(zeroline=False),
                          xaxis=dict(zeroline=False),
                          showlegend=True
                          )

            fig = dict(data=data, layout=layout)
            plotlyoff.plot(fig, filename=save_embedding_image_file_path, auto_open=False)

        return kmeans

    def predict(self, oneworm_dataset_over_seghypcenters, cluster_load_file, num_workers=1):
        scipy_kmeans_model = load(cluster_load_file)

        predicted_seghyplabel_to_cluster_dict = {}
        seghyplabel_to_gtlabel_dict = {}
        gt_labels = []
        seghyp_labels = []  # this basically keeps index values for the other lists, since we are skipping some values
        avg_embeddings_per_seghyp = []

        self.model.eval()
        data_loader = torch.utils.data.DataLoader(oneworm_dataset_over_seghypcenters, shuffle=False, num_workers=num_workers)
        for n_seghyp, seghyp in enumerate(data_loader):
            if torch.any(seghyp['mask'] == -1):  # Bcuz worm18 for 4 con-seghyps behaves bad
                logger.debug('Skipped successfully by checking -1 in mask - n-seghyp was {}'.format(n_seghyp))
                continue
            with torch.no_grad():
                raw = seghyp['raw'].float().cuda()
                mask = seghyp['mask'].float().cuda()
                gt_labels.extend(list(seghyp['gt_label_id'].cpu().numpy()))
                seghyp_labels.append(n_seghyp+1)  # TODO: this only supports batches of 1 for prediction

                batched_embedding = self.model(raw)
                batched_avg_embedding = self.get_avg_embedding_over_mask(batched_embedding, mask)
                avg_embeddings_per_seghyp.extend(
                    [avg_embd.cpu().numpy() for avg_embd in batched_avg_embedding])

        # Compute distances and run hungarian
        avg_embeddings_per_seghyp = np.vstack(avg_embeddings_per_seghyp)
        distance_matrix = scipy_kmeans_model.transform(avg_embeddings_per_seghyp)
        cluster_assignments = linear_sum_assignment(distance_matrix)
        for row_ind, col_ind in zip(*cluster_assignments):
            predicted_seghyplabel_to_cluster_dict.update({seghyp_labels[row_ind]: col_ind+1})
        # This is not required for prediction, but just returning it for easier evaluations
        for sl, gl in zip(seghyp_labels, gt_labels):
            seghyplabel_to_gtlabel_dict.update({sl:gl})
        return predicted_seghyplabel_to_cluster_dict, seghyplabel_to_gtlabel_dict
