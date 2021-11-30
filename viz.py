from bipolar import bipolar
import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap

# get colormap
ncolors = 256
color_array = plt.get_cmap('jet')(range(ncolors))

# change alpha values
color_array[:,-1] = np.linspace(0.0, 1.0, ncolors)

# create a colormap object
map_object = LinearSegmentedColormap.from_list(name='jet_alpha', colors=color_array)

# register this new colormap with matplotlib
plt.register_cmap(cmap=map_object)

plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 14


def show(data, epsr, sources, intensity=False, theme='bright', label=None, saveas=''):
    fig, ax = plt.subplots(1, 1, dpi=100)

    if theme == 'dark':
        colors = 'magma' if intensity else bipolar(neutral=0.0)
        outline = 'white'
        source_color = 'lightyellow'
    else:
        colors = 'jet_alpha' if intensity else 'RdBu'
        outline = 'gray'
        source_color = 'gray'

    ax.imshow(data.T, cmap=colors)
    ax.contour(epsr.T, colors=outline, alpha=0.1)

    for src in sources:
        ax.plot(src[0], src[1], color=source_color)

    ax.invert_yaxis()

    if label:
        offset = int(data.shape[0] / 12)
        ax.text(offset, data.shape[1]-1.5*offset, label, color=outline)

    ax.axis('off')
    plt.tight_layout()
    if saveas:
        plt.savefig(saveas)
        plt.close(fig)
    else:
        plt.show()


def show_design(epsr, *sources):
    fig, ax = plt.subplots(1, 1, dpi=100)

    ax.imshow(epsr.T, cmap='Blues_r')
    for src in sources:
        ax.plot(src[0], src[1], color="red")

    ax.invert_yaxis()

    # Get nice a nice grid going
    ax.minorticks_on()
    ax.grid(which='both', color='white', linestyle='-')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='lightgray')

    # Also show tics on top and right
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    ax.set_xlabel("px")
    ax.set_ylabel("px")

    plt.show()