import numpy as np
from kmeans import Dataset, IdMaker
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import cnames
import random

class DistanceMatrix(object):
    def __init__(self, points):
        self.points = points
        self.matrix = []
        self.coords = None
        self.idmaker = IdMaker()

    def __str__(self):
        labels = [point.label for point in self.points]
        lines = ["\t" + "\t".join(labels)]
        for label, row in zip(self.matrix, labels):
            lines.append("\t".join([label] + row))
        return "\n".join(lines)

    def compute_matrix(self):
        for point1 in self.points:
            self.matrix.append([])
            for point2 in self.points:
                dist = point1.distance(point2)
                self.matrix[-1].append(dist)

    def compute_coords(self):
        matrix = np.matrix(self.matrix)
        self.coords = MDS(dissimilarity='precomputed').fit_transform(matrix)

    def get_colors(self, labels):
        pass

    def set_idmaker(self, idmaker):
        self.idmaker = idmaker

    def plot(self, filename, labels):
        #obtain coordinates
        x = [coord[0] for coord in self.coords]
        y = [coord[1] for coord in self.coords]

        #obtain classes
        classes = [key for key, value in self.idmaker.items.iteritems()]
        all_colours = random.sample(list(cnames.keys()), len(classes)+1)
        #translate class to color
        colours = [all_colours[label] for label in labels]
        plt.scatter(x, y, c=colours)

        #add legend
        recs = []
        for label in classes:
            recs.append(mpatches.Rectangle((0,0),1,1,color=all_colours[self.idmaker.items[label]]))
        plt.legend(recs,classes,loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':10})

        plt.savefig(filename, bbox_inches='tight')
        plt.clf()
