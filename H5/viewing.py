import sys
import matplotlib

# add WatChMaL repository directory to PYTHONPATH
sys.path.append('/home/zhihao/WatChMaL')
# we will use the WatChMaL event display and analysis code
import analysis.event_display.cnn_mpmt_event_display as display

data_path = "/home/zhihao/Data/WCTE_data_fixed/wcte_CDS_pgun_e-_3M_mu-_3M_0to1GeV_fixedFC.h5"
mpmt_positions_filename = "/home/zhihao/Data/WCTE_data_fixed/WCTE_mPMT_image_positions_v3.npz"
geo_filename = "/home/zhihao/Data/WCTE_data_fixed/geofile_wcte.npz"

event_display = display.CNNmPMTEventDisplay(h5file=data_path, mpmt_positions_file=mpmt_positions_filename, geometry_file=geo_filename, channels=["charge","time"], use_new_mpmt_convention=True)

colors={"color_norm": matplotlib.colors.LogNorm(), "color_map": matplotlib.pyplot.cm.turbo}

event_id_to_plot = 42
title = f"Electron event #{event_id_to_plot}"
fig, ax = event_display.plot_event_2d(event_id_to_plot, channel="charge", color_label="Charge", title=title, color_norm=matplotlib.colors.LogNorm(), color_map=matplotlib.pyplot.cm.turbo)
fig.savefig(f"event_{event_id_to_plot}_charge.png")


fig, ax = event_display.plot_event_2d(event_id_to_plot, channel="time", color_label="Time", title=title, color_norm=matplotlib.colors.LogNorm(), color_map=matplotlib.pyplot.cm.turbo)
fig.savefig(f"event_{event_id_to_plot}_time.png")

