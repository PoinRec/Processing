import sys
import matplotlib
import argparse
import os

# add WatChMaL repository directory to PYTHONPATH
sys.path.append('/home/zhihao/WatChMaL')
# we will use the WatChMaL event display and analysis code
import analysis.event_display.cnn_mpmt_event_display as display

parser = argparse.ArgumentParser(description="View an event from HDF5 data with CNNmPMTEventDisplay")
parser.add_argument("data_path", type=str, help="Path to the HDF5 file")
parser.add_argument("mpmt_positions_filename", type=str, help="Path to the mPMT positions .npz file")
parser.add_argument("geo_filename", type=str, help="Path to the geometry .npz file")
parser.add_argument("event_id_to_plot", type=int, nargs="?", default=42, help="Event ID to plot (default: 42)")
args = parser.parse_args()

data_path = args.data_path
mpmt_positions_filename = args.mpmt_positions_filename
geo_filename = args.geo_filename
event_id_to_plot = args.event_id_to_plot

# Create output directory for saving figures
output_dir = os.path.join(os.path.dirname(data_path), "EventPics")
os.makedirs(output_dir, exist_ok=True)

event_display = display.CNNmPMTEventDisplay(h5file=data_path, mpmt_positions_file=mpmt_positions_filename, geometry_file=geo_filename, channels=["charge","time"], use_new_mpmt_convention=True)

colors={"color_norm": matplotlib.colors.LogNorm(), "color_map": matplotlib.pyplot.cm.turbo}

title = f"Event #{event_id_to_plot}"
fig, ax = event_display.plot_event_2d(event_id_to_plot, channel="charge", color_label="Charge", title=title, color_norm=matplotlib.colors.LogNorm(), color_map=matplotlib.pyplot.cm.turbo)
fig.savefig(os.path.join(output_dir, f"event_{event_id_to_plot}_charge.png"))


fig, ax = event_display.plot_event_2d(event_id_to_plot, channel="time", color_label="Time", title=title, color_norm=matplotlib.colors.LogNorm(), color_map=matplotlib.pyplot.cm.turbo)
fig.savefig(os.path.join(output_dir, f"event_{event_id_to_plot}_time.png"))
