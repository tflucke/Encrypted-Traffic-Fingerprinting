#!/usr/bin/python2

import feature_extraction as fe
import itertools

if __name__ == "__main__":
    allTraces = fe.window_all_traces(fe.load_traces())
    print("Saving %d traces to full_traces.dat" % len(allTraces))
    #allTraces = fe.load_pickled_traces("tor", "tor_traces/full_traces.dat")
    fe.pickle_traces(allTraces, "tor_traces/full_traces.dat")
    grouppedTraces = itertools.groupby(allTraces, key=lambda x: x.label)
    maxLen = 9999999999999999999999
    for label, traces in grouppedTraces:
        numTraces = len(list(traces))
        print("Have %d %s traces..." % (numTraces, label))
        maxLen = min(numTraces, maxLen)
    print("Taking the first %d traces of each label" % maxLen)
    grouppedTraces = itertools.groupby(allTraces, key=lambda x: x.label)
    narrowedTraces = []
    for label, traces in grouppedTraces:
        narrowedTraces.extend(list(traces)[0:maxLen])
    print("Saving %d traces to narrowed_traces.dat" % len(narrowedTraces))
    fe.pickle_traces(narrowedTraces, "tor_traces/narrowed_traces.dat")
