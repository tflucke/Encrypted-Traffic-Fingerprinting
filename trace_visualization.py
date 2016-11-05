from trace import *
from feature_extraction import *
import numpy as np
from time import strftime
import matplotlib.pyplot as plt

if __name__ == "__main__":
	all_traces = load_traces()

	Skype = [x for x in all_traces if x.label == 'Skype']
	Torrent = [x for x in all_traces if x.label == 'Torrent']
	Http = [x for x in all_traces if x.label == 'HTTP']
	YouTube = [x for x in all_traces if x.label == 'Youtube']

	for i in Skype:
		H, xedges, yedges = generate_histogram(i)
		X, Y = np.meshgrid(xedges, yedges)
		plt.pcolormesh(X, Y, H)
		plt.show()

	for i in Torrent:
		H, xedges, yedges = generate_histogram(i)
		X, Y = np.meshgrid(xedges, yedges)
		plt.pcolormesh(X, Y, H)
		plt.show()

	for i in Http:
		H, xedges, yedges = generate_histogram(i)
		X, Y = np.meshgrid(xedges, yedges)
		plt.pcolormesh(X, Y, H)
		plt.show()

	for i in YouTube:
		H, xedges, yedges = generate_histogram(i)
		X, Y = np.meshgrid(xedges, yedges)
		plt.pcolormesh(X, Y, H)
		plt.show()
