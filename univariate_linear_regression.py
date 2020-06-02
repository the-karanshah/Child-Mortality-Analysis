# univariate linear regression used in pipeline 2

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
from itertools import islice
import dask
from scipy import stats
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel


def convert_line_to_record(line, col_num):
    values = line.split(',')
    y = values[-1]
    X = np.array([values[col_num]])
    return LabeledPoint(y, X)


def univariate_linear_regression(data, index, feature):
    dataRecords = data.map(lambda line: convert_line_to_record(line, index))
    test, train = dataRecords.randomSplit(weights=[0.2, 0.8], seed=42)
    model = LinearRegressionWithSGD.train(train, iterations=1, step=0.1)
    print(feature, 'Done Training: \n', model)
    return index, feature, model


def get_p(data, index, feature, model, hypothesis_count):
    dataRecords = data.map(lambda line: convert_line_to_record(line, index))
    test, train = dataRecords.randomSplit(weights=[0.2, 0.8], seed=42)
    predictions = model.predict(test.map(lambda x: x.features))
    labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)
    beta = model.weights[0]

    N = test.count()
    m = 1
    rss = labelsAndPredictions \
       .map(lambda vp: (vp[0] - vp[1])**2) \
       .reduce(lambda x, y: x + y)
    xss = test \
       .map(lambda x: x.features[0]**2) \
       .reduce(lambda x, y: x + y)

    df = N - (m + 1)
    s2 = rss / df
    t_stat = beta / ((s2 / xss) ** 0.5)

    p = (1 - stats.t.cdf(abs(t_stat), df=df)) * 2
    p_corrected = p * hypothesis_count
    print(feature, 'Done Hypothesis Testing: Calculated beta and p-value \n', beta, p_corrected)
    return feature, beta, p_corrected


if __name__ == '__main__':
    input_file = sys.argv[1] # "data/processed2_9.csv"
    sc = SparkContext.getOrCreate()

    features = []
    for i, col in enumerate(sc.textFile(input_file).take(1)[0].split(',')):
        features.append((i, col))

    print('Features And Target Label (Last Field): ')
    print(features)

    data = sc.textFile(input_file) \
        .mapPartitionsWithIndex(lambda idx, it: islice(it, 1, None) if idx == 0 else it)  # Ignore first row

    print('Total Number of records to process: ', data.count())
    print('*' * 20)

    lazy_results = []
    for index, feature in features[:-1]:
        lazy_result = dask.delayed(univariate_linear_regression)(data, index, feature)
        lazy_results.append(lazy_result)

    results = dask.compute(*lazy_results)
    print('Resulting beta values for all univariate linear regressions: \n', results)

    print('*' * 20)
    print('Selecting Top 5 positive features for calculating p-values')

    lazy_ps = []
    top_5_positive_features = sorted(results, key=lambda x: x[2].weights[0], reverse=True)[:5]
    for index, feature, model in top_5_positive_features:
        lazy_p = dask.delayed(get_p)(data, index, feature, model, len(results))
        lazy_ps.append(lazy_p)

    ps = dask.compute(*lazy_ps)
    print('Resulting beta and p-values of top 5 positively correlated features: \n', ps)

    print('*' * 20)
    print('Done!')
