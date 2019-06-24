import math
from mne.viz import plot_topomap
import matplotlib.pyplot as plt

def get_topography():
    '''
    Function returenss informantion about channel order and topography
    :return: Dict {Ch_number: [(x,y), Ch_name]}
    '''
    with open('/home/likan_blk/Yandex.Disk/eyelinesOnlineNew/data/order&locations.info', 'r') as f:
        topo_list = [line.split() for line in f][1:]
    topo_dict = {}
    ch_coordinates = []
    ch_names = []
    for elem in topo_list:
        alpha, r = float(elem[1]), float(elem[2])
        alpha = math.pi * alpha / 180.  # to radians
        x, y = r * math.sin(alpha), r * math.cos(alpha)
        name = str(elem[3])
        # topo_dict[int(elem[0])] = [(x, y), name]
        ch_coordinates.append((x,y))
        ch_names.append(name)
    return ch_coordinates,ch_names

def plot_head(data,topography,ch_names,font_size=14):
    # matplotlib.rcParams.update({'font.size': font_size})
    ax = plt.gca()
    plot_topomap(data=data, pos=topography, contours=0, names=ch_names,
                 show_names=True,show=False,axes=ax)
    for text in ax.texts:
        text.set_fontsize(font_size)
    plt.show()
