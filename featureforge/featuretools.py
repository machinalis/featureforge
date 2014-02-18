# -*- coding: utf-8 -*-
from collections import defaultdict
from itertools import chain

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KernelDensity
import numpy
import pylab

from featureforge.featurevectorizer import FeatureMappingVectorizer


class NoTrainDataError(Exception):
    pass


def _build_train(train, feature):
    name = feature.name
    xs = []
    ctr = []
    misses = 0
    for media in train:
        try:
            value = feature(media)
        except ValueError:
            misses += 1
            continue
        xs.append({name: value})
        ctr.append(media["CTR"])
    if not xs:
        raise NoTrainDataError("Train data was empty")
    assert all(isinstance(x, (int, float)) for x in ctr)
    return xs, ctr, misses


def get_feature_stats(feature, corpus, sample_size=1000):
    train, test = corpus(limit=sample_size)
    name = feature.name
    xs, ctr, misses = _build_train(train, feature)

    vectorizer = FeatureMappingVectorizer()
    ys = vectorizer.fit_transform(xs)

    if isinstance(xs[0][name], (int, float)):
        d = {
            "type": "scalar",
            "max": ys.max(),
            "min": ys.min(),
            "mean": ys.mean(),
            "std": ys.std(),
        }
        filtered_ys = []
        filtered_ctr = []
        ylow = d["mean"] - 3 * d["std"]
        yhigh = d["mean"] + 3 * d["std"]
        for y, c in zip(ys, ctr):
            if y >= ylow and y <= yhigh:
                filtered_ys.append(y)
                filtered_ctr.append(c)
        d["raw_feature"] = filtered_ys
        d["raw_ctr"] = filtered_ctr
        d["outliers"] = len(ys) - len(filtered_ys)
        d["correlation"] = numpy.corrcoef(ys.ravel(), ctr)[0, 1]
    elif isinstance(xs[0][name], basestring):
        freq = defaultdict(list)
        for x, y in zip(xs, ctr):
            freq[x[name]].append(y)
        most = max(freq, key=lambda x: len(freq.get(x)))
        least = min(freq, key=lambda x: len(freq.get(x)))
        d = {
            "type": "enumerated",
            "cardinality": len(freq),
            "most_frequent": most,
            "least_frequent": least,
            "most_frequent_proportion": len(freq[most]) / float(len(xs)),
            "least_frequent_proportion": len(freq[least]) / float(len(xs)),
            "freqs": dict(freq),
        }
    else:
        d = {"type": "array"}
    d["name"] = name
    d["trainsize"] = len(xs)
    d["testsize"] = len(list(test))
    d["invalid"] = misses

    regressor = LinearRegression().fit(ys, ctr)

    if len(ctr) < 10:
        d["r2"] = 0
    else:
        d["r2"] = regressor.score(ys, ctr)

    d["regressor"] = regressor
    return d


def evaluate_density_estimation(evidence, samples, bandwidth=0.07):
    de = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    evidence = numpy.array(evidence).reshape(-1, 1)
    de.fit(evidence)
    return numpy.exp(de.score_samples(samples.reshape((-1, 1))))


def plot_stats(stats, ignore_cardinality=False):
    if stats["type"] == "scalar":
        pylab.figure(figsize=(20, 7))
        N = 20
        colors = [pylab.get_cmap("Paired")(0.50)]

        pylab.subplot(122)
        pylab.ylabel("Raw counts")
        pylab.xlabel("{} value".format(stats["name"]))
        data = numpy.array(stats["raw_feature"])
        counts, _, _ = pylab.hist(data, N, color=colors, alpha=0.5)
        X = numpy.linspace(data.min(), data.max(), 100)
        Y = evaluate_density_estimation(data, X, bandwidth=(X[1] - X[0]) * 4)
        Y = Y * (counts.max() / Y.max()) * 0.95
        color = pylab.get_cmap("Paired")(0.99)
        pylab.plot(X, Y, color=color, linewidth=2)

        pylab.subplot(121)
        pylab.ylabel("CTR")
        pylab.xlabel("{} value".format(stats["name"]))
        pylab.ylim(0, 0.2)
        pylab.scatter(stats["raw_feature"], stats["raw_ctr"], color=colors,
                      alpha=0.25)
        Y = stats["regressor"].predict(X.reshape((-1, 1)))
        pylab.plot(X, Y, color=color, linewidth=2.5, linestyle="--")
        pylab.axhline(y=0.01, color=color, alpha=0.5, linestyle="--")
        pylab.show()

    elif stats["type"] == "enumerated":
        scaled_to_max = True
        labels, values = zip(*sorted(stats["freqs"].iteritems()))
        colors = pylab.get_cmap("Paired")(numpy.linspace(0, 1, len(values)))
        if stats["cardinality"] <= 6 or ignore_cardinality:
            pylab.figure(figsize=(20, 7))
            pylab.ylabel("Raw counts")
            pylab.xlabel("CTR")
            pylab.xlim(0, 0.2)
            bincounts, _, _ = pylab.hist(values, 20, color=colors, alpha=0.5)
            data = numpy.array(list(chain(*values)))
            xmin = 0    # data.min()
            xmax = 0.2  # data.max()
            X = numpy.linspace(xmin, xmax, 100)
            if scaled_to_max:
                scale = max(counts.max() for counts in bincounts)
            for label, evidence, counts, color in zip(labels, values,
                                                      bincounts, colors):
                Y = evaluate_density_estimation(evidence, X, 0.01)
                if scaled_to_max:
                    Y = Y * (scale / Y.max()) * 0.95
                else:
                    Y = Y * (counts.max() / Y.max()) * 0.95
                pylab.plot(X, Y, label=label, color=color, linewidth=2,
                           alpha=0.9)
            pylab.legend()
            pylab.show()

        if stats["cardinality"] <= 20 or ignore_cardinality:
            pylab.figure(figsize=(10, 7))
            ypos = numpy.arange(len(labels)) * -1
            X = [len(value) for value in values]
            pylab.barh(ypos, X, align="center", alpha=0.5, color=colors)
            pylab.yticks(ypos, labels)
            pylab.xlabel("Raw counts")
            pylab.show()

    elif stats["type"] == "array":
        pass
    else:
        raise ValueError("Unrecognized stats type")


def judge_r2(x):
    if x > 0.9:
        return "awesome!"
    elif x > 0.75:
        return "good"
    elif x > 0.5:
        return "ok"
    elif x > 0.25:
        return "not good"
    elif x > 0.0:
        return "bad"
    return "pathetic"


def report_feature(feature, corpus, sample_size=2500, ignore_cardinality=False):
    stats = get_feature_stats(feature, corpus, sample_size=sample_size)
    print
    print "Summary"
    print "======="
    print
    print u"Name / type:             {} / {}".format(stats["name"],
                                                     stats["type"])
    print u"R2            :          {: .2f} ({})".format(stats["r2"],
                                                         judge_r2(stats["r2"]))
    print u"Train / invalid / test:   {} / {} / {}".format(stats["trainsize"],
                                                           stats["invalid"],
                                                           stats["testsize"])
    if stats["type"] == "scalar":
        print u"Correlation              {: .2f} ({})".format(
                                                         stats["correlation"],
                                           judge_r2(abs(stats["correlation"])))
        print u"min, max:                [{: .4f}, {: .4f}]".format(
                                                    stats["min"], stats["max"])
        print u"mean:                    {:.4f}".format(stats["mean"])
        print u"standard deviation:      {:.4f}".format(stats["std"])
        print u"outliers:                {}".format(stats["outliers"])
    elif stats["type"] == "enumerated":
        print u"cardinality:             {}".format(stats["cardinality"])
        print u"most frequent label:     {} ({:.2f}%)".format(
               stats["most_frequent"], stats["most_frequent_proportion"] * 100)
        print u"least frequent label:    {} ({:.2f}%)".format(
             stats["least_frequent"], stats["least_frequent_proportion"] * 100)

    plot_stats(stats, ignore_cardinality)
