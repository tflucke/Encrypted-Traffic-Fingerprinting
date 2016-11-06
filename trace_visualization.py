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

	overall_range = determine_histogram_edges(all_traces)

	for i in Skype:
		H, xedges, yedges = generate_histogram(i, overall_range)
		X, Y = np.meshgrid(xedges, yedges)
		plt.xlabel('log(packetsizes)')
		plt.ylabel('log(IAT)')
		plt.title('Skype')
		plt.pcolormesh(X, Y, H)
		plt.colorbar()
		plt.show()

	for i in Torrent:
		H, xedges, yedges = generate_histogram(i, overall_range)
		X, Y = np.meshgrid(xedges, yedges)
		plt.xlabel('log(packetsizes)')
		plt.ylabel('log(IAT)')
		plt.title('Torrent')
		plt.pcolormesh(X, Y, H)
		plt.colorbar()
		plt.show()

	for i in Http:
		H, xedges, yedges = generate_histogram(i, overall_range)
		X, Y = np.meshgrid(xedges, yedges)
		plt.xlabel('log(packetsizes)')
		plt.ylabel('log(IAT)')
		plt.title('Http')		
		plt.pcolormesh(X, Y, H)
		plt.colorbar()
		plt.show()

	for i in YouTube:
		H, xedges, yedges = generate_histogram(i, overall_range)
		X, Y = np.meshgrid(xedges, yedges)
		plt.xlabel('log(packetsizes)')
		plt.ylabel('log(IAT)')
		plt.title('YouTube')		
		plt.pcolormesh(X, Y, H)
		plt.colorbar()
		plt.show()
