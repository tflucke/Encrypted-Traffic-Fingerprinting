from scapy.all import *
import sys
import numpy as np


# The IP from the local computer, for the unencrypted traces : '10.200.1.2'
# For the ipsec encrypted traces: '192.168.1.2'

class Trace():
	"""This class represents a pcap network trace as two lists and a label"""

	# very low constant to avoid 0 IAT
	epsilon = 0.000000001


	def __init__(self,ip="no ip"):
		self.label = ""
		self.timestamps = []
		self.packetsizes = []
		self.num_packets = 0
		self.local_ip = ip

	def load_pcap(self, pathname, label):
		self.label = label
		trace = rdpcap(pathname)
		trace = sorted(trace, key=lambda ts: ts.time)
		for i in range(0, len(trace)):
			if "IP" in trace[i]:
				self.timestamps.append(trace[i].time)
				if trace[i]['IP'].src == self.local_ip:
					self.packetsizes.append(len(trace[i]))
				else:
					self.packetsizes.append(-1*len(trace[i]))
		self.num_packets = len(self.packetsizes)

	def construct_trace(self, packets, times, label):
		self.packetsizes = packets
		self.timestamps = times
		self.label = label
		self.num_packets = len(packets)

	def get_timestamps(self):
		return self.timestamps

	def get_packetsizes(self):
		return self.packetsizes

	# Get the IAT array of a trace
	def get_IAT(self):
		IAT = [self.epsilon]
		for x in range(1, self.num_packets):
			t = self.timestamps[x]-self.timestamps[x-1]
			if t>0:
				IAT.append(t)
			else:
				IAT.append(self.epsilon)
		return IAT
	
	# Get the burst size (bytes) and times arrays of a trace
	def get_burst_info(self):
		burst_size_bytes = []
		burst_size_time = []
		last_direction = np.sign(self.packetsizes[0])
		burst_start = self.timestamps[0]
		burst_stop = 0
		burst_bytes = self.packetsizes[0]
		
		for x in range(1, self.num_packets):
			if last_direction == np.sign(self.packetsizes[x]):
				burst_bytes = last_direction*(abs(burst_bytes)+abs(self.packetsizes[x]))
				if x == self.num_packets-1:
					burst_stop = self.timestamps[x]
					if burst_stop - burst_start > 0:
						burst_size_time.append(burst_stop - burst_start)
					else:
						burst_size_time.append(self.epsilon)
					burst_size_bytes.append(burst_bytes)
			else:
				last_direction = np.sign(self.packetsizes[x])
				burst_stop = self.timestamps[x-1]
				if burst_stop - burst_start > 0:
					burst_size_time.append(burst_stop - burst_start)
				else:
					burst_size_time.append(self.epsilon)
				burst_start = self.timestamps[x]
				burst_size_bytes.append(burst_bytes)
				burst_bytes = self.packetsizes[x]
		return burst_size_bytes, burst_size_time


	# Get a list of windowed traces
	def get_windowed(self, window_size = 1024):

		windowed = []
		packets = self.get_packetsizes()
		times = self.get_timestamps()

		for x in range(0,(self.num_packets/window_size)):
			temp = Trace(self.local_ip)
			temp.construct_trace(packets[x*window_size:(x+1)*window_size], times[x*window_size:(x+1)*window_size], self.label)
			windowed.append(temp)

		return windowed

