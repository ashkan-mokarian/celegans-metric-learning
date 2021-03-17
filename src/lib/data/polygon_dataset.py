import numpy as np
import cv2

import torch
from torch.utils.data import Dataset


class PolygonDataset(Dataset):
    def __init__(self, mode='train', n_shapes=4, data_size=512):
        super().__init__()
        self.mode = mode
        self.n_shapes = n_shapes
        self.data_size = data_size
        self.height = 256
        self.width = 256

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        while True:
            img = np.ones((self.height, self.width), dtype=np.uint8) * 255
            shapes = np.zeros((0, self.height, self.width), dtype=np.uint8)
            for ncorner in range(3, self.n_shapes+3):
                x = np.random.randint(30, 225)
                y = np.random.randint(30, 225)
                r = 15
                corners = np.array([[x+r*np.sin(theta), y+r*np.cos(theta)]
                           for theta in np.linspace(0, 2*np.pi, num=ncorner, endpoint=False)], dtype=np.int32)
                # corners = corners.reshape((-1, 1, 2))
                theta = np.random.randint(-90, 90)
                canvas = np.copy(img)
                # shape = np.int0(cv2.fillPoly(canvas ,corners, color='blue'))
                cv2.fillConvexPoly(canvas, corners, color=(100,100,100))
                cv2.imwrite('./sag.jpg', canvas)

                gt = np.zeros_like(img)
                gt = cv2.fillPoly(gt, [box], 1)
                ins[:, gt != 0] = 0
                ins = np.concatenate([ins, gt[np.newaxis]])
                img = cv2.fillPoly(img, [box], 255)
                img = cv2.drawContours(img, [box], 0, 0, 2)

            # minimum area of stick
            if np.sum(np.sum(ins, axis=(1, 2)) < 400) == 0:
                break

        if self.train:
            sem = np.zeros_like(img, dtype=bool)
            sem[np.sum(ins, axis=0) != 0] = True
            sem = np.stack([~sem, sem]).astype(np.uint8)

            # 1 * height * width
            img = torch.Tensor(img[np.newaxis])
            # 2 * height * width
            sem = torch.Tensor(sem)
            # n_sticks * height * width
            ins = torch.Tensor(ins)
            return img, sem, ins
        else:
            # 1 * height * width
            img = torch.Tensor(img[np.newaxis])
            return img