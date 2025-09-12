# PlotRoA.py  # 2022-11-22
# MIT LICENSE 2020 Ewerton R. Vieira

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib
import os
import csv
import numpy as np

def PlotRoA(lower_bounds, upper_bounds, selection=[], fig_w=8, fig_h=8, xlim=None, ylim=None, fontsize=16,
              cmap=matplotlib.cm.get_cmap('viridis', 256), name_plot="", from_file=None, plot_point=False, section=None, from_file_basic=False, dir_path=""):
    """ TODO:
    * section = ([z,w],(a,b,c,d)), 3D section when [z,w]=(c,d)
    * selection = selection of morse sets
    * check 1D and 3D plottings
    * check save file"""

    dim = len(lower_bounds)

    eps_round_up = 0 # 0.000000005 parameter that remove white lines from the plot for some cases.

    # path to save and read files
    if not dir_path:
        dir_path = os.path.join(os.getcwd(), "output")

    variables = [a for a in range(dim)]

    morse = {}  # tiles in morse sets
    tiles = {}  # tiles in regions of attraction (not including tiles in morse set)

    # read file saved by RoA
    if from_file and not from_file_basic:
        from_file = os.path.join(dir_path, from_file + "_RoA_.csv")
        with open(from_file, "r") as file:
            f = csv.reader(file, delimiter=',')
            next(f)
            box_size = [float(i) for i in next(f)[0:dim]]
            next(f)
            Tiles = []
            Morse_nodes = []
            Boxes = []
            num_morse_sets = 0
            counter_temp = 0
            for row in f:
                if row[0] == "Tile_in_Morse_set":
                    counter4morse_sets = counter_temp
                    continue
                counter_temp += 1
                Tiles.append(int(row[0]))
                Morse_nodes.append(int(row[1]))
                Boxes.append([float(a) for a in row[2:2+2*dim]])
                if num_morse_sets < int(row[1]):   # find the num_morse_sets - 1
                    num_morse_sets = int(row[1])
            num_morse_sets += 1

        if not selection:
            selection = [i for i in range(num_morse_sets)]

        cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_morse_sets-1)

        volume_cube = 1
        d_vol = dict()
        for i in range(dim):
            volume_cube *= box_size[i]

        for i, m_node in enumerate(Morse_nodes):
            if m_node not in selection:  # only add the selected Morse sets
                continue
            clr = matplotlib.colors.to_hex(cmap(cmap_norm(m_node)))
            if i < counter4morse_sets:  # associate center of boxes to the Morse tiles
                B = tiles.get(clr, [])
                B.append(Boxes[i])
                tiles[clr] = B
                d_vol[m_node] = d_vol.get(m_node, 0) + volume_cube
            else:  # associate  boxes to the Morse sets
                A = morse.get(clr, [])
                A.append(Boxes[i])
                morse[clr] = A
                d_vol[m_node] = d_vol.get(m_node, 0) + volume_cube

        print(f'dictionary with volume of all Morse tiles = {d_vol}')

    # read file saved by CMGDB (only Morse tiles)
    if from_file and from_file_basic:
        from_file = os.path.join(dir_path, from_file + ".csv")
        morse = {}
        with open(from_file, "r") as file:
            f = csv.reader(file, delimiter=',')
            Morse_nodes = []
            box = []
            for row in f:
                dim = len(row)//2
                Morse_nodes.append(int(float(row[-1])))
                box.append([float(row[i]) for i in range(2*dim)])
            box_size = [float(row[i+dim]) - float(row[i]) + eps_round_up for i in range(dim)]

        num_morse_sets = Morse_nodes[-1] + 1
        cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_morse_sets-1)
        for i, m_node in enumerate(Morse_nodes):
            clr = matplotlib.colors.to_hex(cmap(cmap_norm(m_node)))
            A = morse.get(clr, [])
            A.append(box[i])
            morse[clr] = A
        tiles = morse

    # for dim 1, add fake dimension
    if dim == 1:
        size1d = 64
        box_size.append(fig_h/size1d)
        lower_bounds += lower_bounds
        upper_bounds += upper_bounds
        variables = [0,1]

    # 2D plotting or 2D with a given section
    if section or dim <= 2:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        if dim <= 2:
            section = ([], 'projection')
        variables_section = list(set(variables) - set(section[0]))
        d1 = variables_section[0]
        d2 = variables_section[1]
        if section[1] == 'projection':
            section = ([], 'projection')  # clean section to do projection

        if not from_file_basic:
            for i, j in tiles.items():
                rectangles_list = []
                for row in j:
                    if section[1] == 'projection':
                        in_section = [True]
                    else:
                        in_section = [row[k] - box_size[k]/2 <= section[1][k]
                                      < row[k] + box_size[k]/2 for k in section[0]]
                    if all(in_section):
                        box_size = [float(row[i+dim]) - float(row[i]) + eps_round_up for i in range(dim)]
                        if dim == 1:
                            box_size.append(fig_h/size1d)
                            row[d2] = -fig_h/(2*size1d)
                        rectangle = Rectangle((row[d1], row[d2]), box_size[d1], box_size[d2])
                        rectangles_list.append(rectangle)
                pc = PatchCollection(rectangles_list, cmap=cmap, fc=i, alpha=0.4, ec='none')
                ax.add_collection(pc)
        for i, j in morse.items():
            rectangles_list = []
            for row in j:
                if section[1] == 'projection':
                    in_section = [True]
                else:
                    in_section = [row[k] - box_size[k]/2 <= section[1][k]
                                  < row[k] + box_size[k]/2 for k in section[0]]
                if all(in_section):
                    box_size = [float(row[i+dim]) - float(row[i]) + eps_round_up for i in range(dim)]
                    if dim == 1:
                        box_size.append(fig_h/size1d)
                        row[d2] = -fig_h/(2*size1d)
                    rectangle = Rectangle((row[d1], row[d2]), box_size[d1], box_size[d2])
                    rectangles_list.append(rectangle)
            pc = PatchCollection(rectangles_list, cmap=cmap, fc=i, alpha=1, ec='none')
            ax.add_collection(pc)
        tick = 5  # tick for 2D plots
        if xlim and ylim:
            ax.set_xlim([xlim[0], xlim[1]])
            ax.set_ylim([ylim[0], ylim[1]])
            plt.xticks(np.arange(xlim[0], xlim[1], tick))
            plt.yticks(np.arange(ylim[0], ylim[1], tick))
            plt.xticks(np.linspace(xlim[0], xlim[1], tick))
            plt.yticks(np.linspace(ylim[0], ylim[1], tick))
        else:
            ax.set_xlim([lower_bounds[d1], upper_bounds[d1]])
            ax.set_ylim([lower_bounds[d2], upper_bounds[d2]])
            plt.xticks(np.arange(lower_bounds[d1], upper_bounds[d1], tick))
            plt.yticks(np.arange(lower_bounds[d2], upper_bounds[d2], tick))
            plt.xticks(np.linspace(lower_bounds[d1], upper_bounds[d1], tick))
            plt.yticks(np.linspace(lower_bounds[d2], upper_bounds[d2], tick))
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        ax.xaxis.label.set_size(fontsize)
        ax.yaxis.label.set_size(fontsize)
        ax.set_xlabel(str(d1))
        ax.set_ylabel(str(d2))
        if section[1] == 'projection':
            value_section = tuple([0 for i in section[0]])
        else:
            value_section = tuple([int(section[1][i]*100) for i in section[0]])
        name_plot = f'{dir_path}{name_plot}'
        name_plot += f'_{section[0]}_{value_section}' if section[0] else ''
        if name_plot != dir_path:
            plt.savefig(name_plot)
        return fig, ax
    # 3D plotting
    else:
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_subplot(111, projection='3d')
        def cuboid_data(center, size):
            o = [center[i] - size[i]/2 for i in range(3)]
            d = [size[i] for i in range(3)]
            return np.array([
                [o[0],     o[1],     o[2]],
                [o[0]+d[0],o[1],     o[2]],
                [o[0]+d[0],o[1]+d[1],o[2]],
                [o[0],     o[1]+d[1],o[2]],
                [o[0],     o[1],     o[2]+d[2]],
                [o[0]+d[0],o[1],     o[2]+d[2]],
                [o[0]+d[0],o[1]+d[1],o[2]+d[2]],
                [o[0],     o[1]+d[1],o[2]+d[2]],
            ])
        def cuboid_faces(pts):
            return [
                [pts[0], pts[1], pts[2], pts[3]],
                [pts[4], pts[5], pts[6], pts[7]],
                [pts[0], pts[1], pts[5], pts[4]],
                [pts[2], pts[3], pts[7], pts[6]],
                [pts[1], pts[2], pts[6], pts[5]],
                [pts[4], pts[7], pts[3], pts[0]]
            ]
        # Plot tiles (ROA, alpha=0.4)
        for i, j in tiles.items():
            faces_list = []
            for row in j:
                box_size = [float(row[k+3]) - float(row[k]) + eps_round_up for k in range(3)]
                center = [(row[k] + row[k+3])/2 for k in range(3)]
                pts = cuboid_data(center, box_size)
                faces = cuboid_faces(pts)
                faces_list.extend(faces)
            if faces_list:
                pc = Poly3DCollection(faces_list, facecolors=[i], alpha=0.4, edgecolor='none')
                ax.add_collection3d(pc)
        # Plot Morse sets (alpha=1)
        for i, j in morse.items():
            faces_list = []
            for row in j:
                box_size = [float(row[k+3]) - float(row[k]) + eps_round_up for k in range(3)]
                center = [(row[k] + row[k+3])/2 for k in range(3)]
                pts = cuboid_data(center, box_size)
                faces = cuboid_faces(pts)
                faces_list.extend(faces)
            if faces_list:
                pc = Poly3DCollection(faces_list, facecolors=[i], alpha=1, edgecolor='none')
                ax.add_collection3d(pc)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Theta')
        # Always save the 3D plot
        if not name_plot or name_plot == dir_path:
            name_plot = os.path.join(dir_path, "roa_3d_plot.png")
        plt.savefig(name_plot)

        # Always save a 2D slice at z3=0 if 3D is plotted
        try:
            fig2d, ax2d = plt.subplots(figsize=(fig_w, fig_h))
            rectangles_list = []
            # Slice at z3=0, width=0.05
            slice_dim = 2
            slice_val = 0.0
            slice_width = 0.05
            for i, j in tiles.items():
                for row in j:
                    z1_center = (row[0] + row[3]) / 2
                    z2_center = (row[1] + row[4]) / 2
                    z3_center = (row[2] + row[5]) / 2
                    # Diagonal plane: z3 ≈ z1
                    if abs(z3_center - z1_center) < slice_width:
                        rectangle = Rectangle((row[0], row[1]), row[3]-row[0], row[4]-row[1], color=i, alpha=0.4, linewidth=0)
                        rectangles_list.append(rectangle)
            for i, j in morse.items():
                for row in j:
                    z1_center = (row[0] + row[3]) / 2
                    z2_center = (row[1] + row[4]) / 2
                    z3_center = (row[2] + row[5]) / 2
                    if abs(z3_center - z1_center) < slice_width:
                        rectangle = Rectangle((row[0], row[1]), row[3]-row[0], row[4]-row[1], color=i, alpha=1, linewidth=0)
                        rectangles_list.append(rectangle)
            for rect in rectangles_list:
                ax2d.add_patch(rect)
            ax2d.set_xlabel('z1')
            ax2d.set_ylabel('z2')
            ax2d.set_xlim([lower_bounds[0], upper_bounds[0]])
            ax2d.set_ylim([lower_bounds[1], upper_bounds[1]])
            plt.tight_layout()
            plt.savefig(os.path.join(dir_path, "roa_2d_slice.png"))
            plt.close(fig2d)
        except Exception as e:
            print("2D slice saving failed:", e)

        return fig, ax