Nuclei Metric Learning
==============================

Metric learning for worms nuclei.


# Scripts

## consolidate_worms_dataset
Call: `./src/scripts/consolidate_worms_dataset -c default.toml -i path_to_30WormsImagesGroundTruthSeg`

Creates .hdf datasets for each worm. Gets rid of worm names and mis-matched label numbering by unifying these with
 `universe.txt` and `worm_names.txt`. Most setting are set in .toml config file (check default.toml), such as
  `worms_dataset` that here is used as path to output dataset, but throughout the project is used as input path.

**HDF keys**:
- `volumes/raw`: raw input, [140x140x1166] uint8, without any modification e.g. normalization
- `volumes/nuclei_seghyp`: instace labeling, [140x140x1166] uint16, labels no meaning, just a number to
 distinguish between instances
- `matrix/con_seghyp`: center of nuclei, each row corresponds to the label in `volumes/nuclei_seghyp`, [max
(nuclei_instances), 3] float32
- `volumes/gt_nuclei_labels`: ground truth labels, segmentations the same as nuclei_instances, here just the invalid
 segmentations are removed, and also relabeled according to `universe.txt`, [140x140x1166] uint16
- `matrix/gt_con_labels`: same as `con_instances` but for `gt_nuclei_labels`, all fixed size of [559x3] float32
, missing labels are np.array([0.0, 0.0, 0.0])


## consolidate_cpm_dataset
Call: `./src/scripts/consolidate_cpm_dataset -c default.toml -i path_to_root_kolmogorov_sol_format_both_directions
 -i2 path_to_nucleinames_corresponding_to_QAP_sols_labeling -i3
  path_to_nuclei_name_labels_in_30WormsImageGroundTruthInstanceSeg`

Creates cpm dataset, default in ./data/processed (defined in default.toml). .pkl file containing a dictionary with
 keys '{w1id}-{w2id}' where w1id<w2id and value is a dict of consistent pairwise matchings.
# Models
**convnet_models**: **(No Use For Now)** a conventional VGG-style network with some conv layers + some fc layers. For
 extracting embeddings based on patches.
 
 **unet**: unet model for pixel-wise embeddings.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
