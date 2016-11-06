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
		a = plt.pcolormesh(X, Y, H.transpose()[::-1])
		plt.colorbar()
		a.set_clim([0,0.6])
		plt.show()

	for i in Torrent:
		H, xedges, yedges = generate_histogram(i, overall_range)
		X, Y = np.meshgrid(xedges, yedges)
		plt.xlabel('log(packetsizes)')
		plt.ylabel('log(IAT)')
		plt.title('Torrent')
		a = plt.pcolormesh(X, Y, H.transpose()[::-1])
		plt.colorbar()
		a.set_clim([0,0.6])
		plt.show()

	for i in Http:
		H, xedges, yedges = generate_histogram(i, overall_range)
		X, Y = np.meshgrid(xedges, yedges)
		plt.xlabel('log(packetsizes)')
		plt.ylabel('log(IAT)')
		plt.title('Http')		
		a = plt.pcolormesh(X, Y, H.transpose()[::-1])
		plt.colorbar()
		a.set_clim([0,0.6])
		plt.show()

	for i in YouTube:
		H, xedges, yedges = generate_histogram(i, overall_range)
		X, Y = np.meshgrid(xedges, yedges)
		plt.xlabel('log(packetsizes)')
		plt.ylabel('log(IAT)')
		plt.title('YouTube')		
		a = plt.pcolormesh(X, Y, H.transpose()[::-1])
		plt.colorbar()
		a.set_clim([0,0.6])
		plt.show()
