"""used to cluster group in a hdf5 files
"""

import sys
from h5reader import H5Reader
from kmeans import HKMeans, KMeans, Dataset
from matrix import DistanceMatrix
import pickle
import os.path

def get_file_names():
    points = {}
    with open("test_bin701_v1.txt", 'r') as input_file:
        for line in input_file:
            line = line.split()
            points[line[0]] = (line[1], " ".join(line[2:]))
    return points

def get_points():
    h5reader = H5Reader(FILE_PATH)
    points = []
    for filename, label in get_file_names().iteritems():
        data = h5reader.get_dset(filename)
        point = Dataset(data, name=filename, label=label[0], comment=label[1])
        points.append(point)
    return points

def evaluate_clustering(clustering_result, name, iterations):
    """evalutes the clusters for a clustering function
    clustering_result is a ClusteringAlgorithm
    """
    aris = []
    homs = []
    comps = []
    for _ in range(iterations):
        kmeans = clustering_result
        ari, hom, comp = kmeans.eval_clusters()
        aris.append(ari)
        homs.append(hom)
        comps.append(comp)
    avg_ari = sum(aris)/float(len(aris))
    avg_hom = sum(homs)/float(len(homs))
    avg_comp = sum(comps)/float(len(comps))
    #avg_ari = max(aris)
    #avg_hom = max(homs)
    #avg_comp = max(comps)
    print '{0}:\n\tAdjusted Rand Index: {1}' \
          '\n\tHomogeneity: {2}' \
          '\n\tCompleteness: {3}'.format(name, avg_ari, avg_hom, avg_comp)

def main():
    """reads a hdf5 file and clusters it's datasets
    """
    #compare()
    #print "KMeans"
    #print run_kmeans()
    hkmeans = HKMeans(get_points(), K_CLUSTERS)
    hkmeans.generate_clusters()
    points, labels_true, labels_pred = hkmeans.get_labels()
    matrix = DistanceMatrix(points)
    matrix.compute_matrix()
    matrix.compute_coords()

    matrix.set_idmaker(hkmeans.idmaker)
    matrix.plot('true.png', labels_true)
    matrix.plot('pred.png', labels_pred)

if __name__ == '__main__':
    FILE_PATH = sys.argv[1]
    K_CLUSTERS = int(sys.argv[2])
    main()
