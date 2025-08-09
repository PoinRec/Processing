# Processing
A personal collection of scripts for converting, visualizing, and evaluating models in the WatChMaL framework.

All scripts need same environment as the **WatChMaL** repo, so maybe you need to run them in a container.



## Evaluation
Scripts for visualization and valuation of trained model.


### Usage (Except for `table_plot.py`)
```bash
*.py -r path/to/run/dir
```

Specially, for scripts associated with classification, like `classification.py` and `FC_*.py`, you can add an extra argument `-e` to adjust the mis-identification rate you want for plots and the output files.

### `table_plot.py`

Scripts to generate an image of a table helping you quickly summarize the performance of the model you trained.


## H5

Scripts for viewing, checking and splitting the H5 data files for training.

- `checking.py`: a script to check your H5 files are normal, especially for the label *fully_contained*, which should be converted into bool dtype for FC classification (done in `converting.py`)

- `converting.py`: a script to convert your H5 files so it can be used for training of FC classification, as mentioned above.

-  `distribution.py`: a script to plot the data count distribution ersus various different labels.

- `FCtest.py`: a script to plot the FC dependence of energy (its existence is due to some *historical* reason).

- `split4*.py`: different splitting scripts for different purpose.

- `viewing.py`: a script to view a specific event on the unfolded image.



## Mapping

Scripts for checking if your mapping is correct. The purpose of the scripts are all on their names.


## Tools

Toolkit, in case you need them some day.

