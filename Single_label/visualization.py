from trace import *
from feature_extraction import *
import numpy as np
import sys
from time import strftime
import matplotlib.pyplot as plt
import numpy.random as nprnd
import argparse as ap

FEATURES = ['size_IAT', 'burst', 'rtt']
modes = ['tor','unencr','ipsec_ns','ipsec_def','ipsec_20ps','ipsec_50ps','ipsec_100ps','ipsec_150ps','ipsec_200ps','ipsec_300ps','ipsec_400ps','ipsec_20_ud','ipsec_50_ud','ipsec_100_ud','ipsec_150_ud','ipsec_200_ud','ipsec_300_ud','ipsec_400_ud']

if __name__ == "__main__":
	
    parser = ap.ArgumentParser(description = "Create visualizations from data")
    parser.add_argument('mode', metavar = 'mode', type = str,
                        action = 'store', help = 'The mode to use.')
    parser.add_argument('--trace', dest = 'trace_file')
    parser.add_argument('--show', action = 'store_true',
                        help = 'Display figures after saving them to file.')
    args = parser.parse_args()

    # Exit in invalid mode.
    if mode not in modes:
        err_str = "Invalid mode: {}. Execute as '{} [mode]' for modes {}"
        sys.exit(err_str.format(mode, parser.prog, modes))

    # Load traces from trace file.
    if args.trace_file:
        try:
            all_traces = load_pickled_traces(load_mode=mode, o_file=args.trace_file)
        except IOError:
            sys.exit("Could not open trace file '{}'".format(args.trace_file))
    else:
        try:
            all_traces = load_pickled_traces(load_mode=mode)
        except IOError:
            sys.exit("Could not open default trace file")

    # Consolidate traces for each label.
    all_traces = consolidate_traces(all_traces)
    for feature in FEATURES:
        overall_range = determine_histogram_edges(feature, all_traces)

        # Plot all traces as a whole
        for i in all_traces:
            if args.trace_file:
                filename = args.trace_file.split('/')[1][:-4] + '_'
            else:
                filename = 'default'

            if feature == 'size_IAT':
                H, xedges, yedges = generate_histogram_size_IAT(i, overall_range)
                plt.xlabel('log(packetsizes)')
                plt.ylabel('log(IAT)')
                plt.title(str(i.label) + ' IAT')
                filename += str(i.label) + '_sizeIAT.png'
            elif feature == 'burst':
                H, xedges, yedges = generate_histogram_burst(i, overall_range)
                plt.xlabel('log(burst size (bytes))')
                plt.ylabel('log(burst time)')
                plt.title(str(i.label) + ' Burst')
                filename += str(i.label) + '_burst.png'
            elif feature == 'rtt':
                H, xedges, yedges = generate_histogram(feature, i, overall_range)
                plt.xlabel('log(RTT)')
                plt.ylabel('Service')
                plt.title(str(i.label) + ' RTT')
                filename += str(i.label) + '_rtt.png'

            X, Y = np.meshgrid(xedges, yedges)
            a = plt.pcolormesh(X, Y, H.transpose()[::-1])
            a.set_clim([0,0.6] if feature == 'burst' else [0, 0.15])
            plt.colorbar()
            plt.savefig(filename)
            if args.show:
                plt.show()
