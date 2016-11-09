from scapy.all import *
import sys

class Trace():
	"""This class represents a pcap network trace as two lists and a label"""

	# The IP from the local computer
	local_ip = '10.200.1.2'

	# very low constant to avoid 0 IAT
	epsilon = 0.000000001
	window = 1000


	def __init__(self):
		self.label = ""
		self.timestamps = []
		self. packetsizes = []
		self.num_packets = 0

	def load_pcap(self, pathname, label):
		self.label = label
		trace = rdpcap(pathname)
		trace = sorted(trace, key=lambda ts: ts.time)
		for i in range(1, len(trace)):
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
	
	# Get a list of windowed traces
	def get_windowed(self):
		windowed = []
		packets = self.get_packetsizes()
		times = self.get_timestamps()

		for x in range(0,(self.num_packets/self.window)):
			temp = Trace()
			temp.construct_trace(packets[x*self.window:(x+1)*self.window], times[x*self.window:(x+1)*self.window], self.label)
			windowed.append(temp)

		return windowed

