import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":8,"axes.labelsize":8,"legend.fontsize":6.5,
  "xtick.labelsize":7,"ytick.labelsize":7,"lines.linewidth":1.0,
  "font.family":"serif","mathtext.fontset":"cm","figure.dpi":150,
  "axes.linewidth":0.6})
def panel(ax,lab):
    ax.text(-0.02,1.03,lab,transform=ax.transAxes,va="bottom",ha="right",fontweight="bold",fontsize=8)
