# multivariate linear regression used in pipeline 2

# EDA for all csvs in one run

# System:
# GCP Cluster Details:
# Master node: Standard (1 master, N workers)
# Machine type: e2-standard-2
# Number of GPUs: 0
# Primary disk type: pd-standard
# Primary disk size: 64GB
# Worker nodes: 4 (2 of which were up and running)
# Machine type: e2-highmem-4
# Number of GPUs: 0
# Primary disk type: pd-standard
# Primary disk size: 32GB
# Image Version: 1.4.27-debian9

import sys
from pyspark import SparkContext
import numpy as np
from scipy import stats
from itertools import islice
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel


def convert_line_to_record(line):
    values = line.split(',')
    y = int(values[-1])
    X = np.array(values)[3:-1]
    return LabeledPoint(y, X)


if __name__ == '__main__':
    input_file = sys.argv[1]  # "data/processed2_9.csv"
    sc = SparkContext.getOrCreate()

    features = []
    print('Features And Target Label (Last Field): ')
    for i, col in enumerate(sc.textFile(input_file).take(1)[0].split(',')):
        features.append((i, col))
    print(features)

    data = sc.textFile(input_file) \
        .mapPartitionsWithIndex(lambda idx, it: islice(it, 1, None) if idx == 0 else it)  # Ignore first row

    dataRecords = data.map(lambda line: convert_line_to_record(line))
    print('Total Number of records to process: ', data.count())
    print('*' * 20)

    model = LinearRegressionWithSGD.train(dataRecords, iterations=1, step=1)
    print('Resulting beta values for all univariate linear regressions: \n', model.weights)

    predictions = model.predict(dataRecords.map(lambda x: x.features))
    labelsAndPredictions = dataRecords.map(lambda lp: lp.label).zip(predictions)

    N = dataRecords.count()
    m = len(model.weights)
    rss = labelsAndPredictions \
        .map(lambda vp: (vp[0] - vp[1]) ** 2) \
        .reduce(lambda x, y: x + y)

    df = N - (m + 1)
    s2 = rss / df

    print('*' * 20)
    print('Calculating p-values for all features')

    for index, feature in enumerate(features[3:-1]):
        print(index, feature)
        beta = model.weights[index]
        xss = dataRecords \
            .map(lambda x: x.features[index] ** 2) \
            .reduce(lambda x, y: x + y)
        t_stat = beta / ((s2 / xss) ** 0.5)
        p = (1 - stats.t.cdf(abs(t_stat), df=df)) * 2
        p_corrected = p * len(model.weights)
        print(feature[1], 'Done Hypothesis Testing: Calculated beta and p-value \n', beta, p_corrected)

    print('*' * 20)
    print('Done!')
