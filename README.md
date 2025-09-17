# Processing
A personal collection of scripts for converting, visualizing, and evaluating models in the WatChMaL framework.

All scripts (except for `Tools/check_root.py` which also needs environment for `pyROOT`) need same environment as the [**WatChMaL** repo](https://github.com/WatChMaL/WatChMaL), so maybe you need to run them in a container that the whole WatChMaL ML process can work in.

## process_all.sh
A `.sh` script that can help you do all possible checkings and viewings when you first get brand-new data. But if you want to use it you must have the same container image `larcv2_ub2204-cuda121-torch251-larndsim-2025-03-20.sif` as me, or you can modify the image's name by yourself in this script.

### usage
Run the below command line:

```bash
./process_all.sh
```
---
If your get some error named `Permission denied`, then you may don't have the permission to execute it. You can either use

```bash
shell ./process_all.sh
```
to run it, or run the command line below to change the mode:

```bash
chmod u+x process_all.sh
```
---
If you have run everything smoothly, you should see
```bash
data path:
geometry path:
image position path:
Event number you want to check:
```
appear in sequence, you should type the absolute paths of these three files or the event number one at a time and press **Enter** each time after you've finish typing. Then you should successfully run it.




## Evaluation
Scripts for visualization and valuation of trained model.


### Usage (Except for `table_plot.py`)
```bash
python *.py path/to/run/dir
```

Specially, for scripts associated with classification, like `classification.py` and `FC_*.py`, you can add an extra argument `-e` to adjust the mis-identification rate you want for plots and the output files.

All the evaluation results will be saved under `results/` under the run directory.

### `table_plot.py`

Scripts to generate an image of a table helping you quickly summarize the performance of the model you trained.

#### Usage: 
```bash
export TABLE_PLOT_DATA=path/to/your/json/file
python table_plot.py
```

There's a template under `Processing/Evaluation/config` that can tell you how to write your own json file. Plus, please name your json file `table_plot_data_*.json` so that it can be ignored by `.gitignore`.

## H5

Scripts for viewing, checking and splitting the H5 data files for training.

All these `.py` scripts needs some argument indicating the path to your `.h5` file.

And for `FCtest.py` you need to pass another argument of the path to the splitting `.npz` file. (For `view.py`, it needs even more arguments. You can check the code for the details.)

### Basic Usage

```bash
python *.py path/to/your/h5/file (Some/potential/extra/path)
```

### Simple introduction of each script

- `checking.py`: a script to check your H5 files are normal, especially for the label `fully_contained`, which should be converted into bool dtype for FC classification (done in `converting.py`)

- `converting.py`: a script to convert your H5 files so it can be used for training of FC classification, as mentioned above (The new `.h5` file will be named same as the old one but begins with `FC_`).

-  `distribution.py`: a script to plot the data count distribution ersus various different labels (The output images will be saved under `Distribution/` which is under the same folder as your `.h5` file).

- `FC*.py`: a script to plot the FC dependence of energy / towall or 2D dependence of both (`FC_check.py`).

- `split4*.py`: different splitting scripts for different purpose (All splitting `.npz` file will be saved under `Splitting/`).

- `viewing.py`: a script to view a specific event on the unfolded image (Pictures will be saved under `EventPics/`).



## MappingCheck

Scripts for checking if your mapping is correct. The purpose of the scripts are all on their names. Results will be saved under `Checking/` which is under the same folder as your geometry file.

### Basic Usage
For `continuity_check.py` and `viewing.py`, you need to pass two arguments that is path to your geometry file and mPMT image position file.

For `orientation_check.py`, only path to the geometry file is needed.

```bash
python *.py path/to/your/geometry/file (Some/potential/extra/path/to/your/mpmt/position/file)
```

### Simple introduction of each script

- `continuity_check.py`: a script to check if the mapping is correct (especially the continuity, that is, the adjacent mPMT should remain adjacent on the image plot).

- `orientation_check.py`: a script to check if the mPMT orientations are correct (You can check if the different colors representing different numbers of PMTs in every mPMT are aligned in the same way or not), which is especially for the WCTE data, since it's confirmed to have wrong orientations.

- `viewing.py`: a script to view the 3D structure of the ditector with numbers next to the mPMTs.


## Tools

Toolkit, in case you need them some day.

- `check_root.py`: check the `.root` files (especially gamma), which has something to do with the primary particles and pair production ($\gamma \longrightarrow e^- + e^+$). When generating $\gamma$, the primary particles for GPS is simply $\gamma$ so it has something wrong when converting the `.root` files into `.npz` files. You should use gamma-conversion mode so that the primary particles is $e^-/e^+$.

- `compare_npz.py`: to compare if two `.npz` files are identical, and to list their difference if they are not identical.

- `compare_yaml.py`: to compare if two `.txt` files are identical (useful especially for comparing hyperparameters in different `.yaml` files), and to list their difference in a `git diff` way if they are not identical.

- `getRH`: a script left over by history, very useful if you really don't know the geometry of some detectors.

- `viewnpz`: to understand the overall structure of a `.npz` file.


### Usage

```bash
python compare_npz.py file_1.npz file_2.npz
```

```bash
python compare_yaml.py file_1.txt file_2.txt
```

```bash
python getRH.py path/to/your/geometry_file.npz
```

```bash
python viewnpz.py 
```
