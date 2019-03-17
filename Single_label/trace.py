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
                self.rtts = []
		self.num_packets = 0
		self.local_ip = ip

	def load_pcap(self, pathname, label):
		self.label = label
                #print "Reading pcap..."
		trace = rdpcap(pathname)
                #print "Sorting pcap..."
		trace = sorted(trace, key=lambda ts: ts.time)
                #print "Processing packets..."
                rtts = {}
		for p in trace:
			if 'IP' in p:
				self.timestamps.append(p.time)
				if p['IP'].src == self.local_ip:
					self.packetsizes.append(len(p))
				else:
					self.packetsizes.append(-1*len(p))
                        # Calculate RTTs.
                        if 'TCP' in p:
                                if p['TCP'].flags == 0x02: # SYN packet.
                                        rtts[p['TCP'].seq] = RTT(p.time, p['TCP'].seq, p['IP'].src, p['IP'].dst)
                                # SYN/ACK packet.
                                elif p['TCP'].flags ==  0x12 and p['TCP'].ack - 1 in rtts:
                                        rtts[p['TCP'].ack - 1].set_ack(p['TCP'].ack)
                                # ACK packet for SYN/ACK.
                                elif p['TCP'].flags ==  0x10 and p['TCP'].seq - 1 in rtts:
                                        rtts[p['TCP'].seq - 1].set_rtt(p.time)

                self.rtts = [rtt for _, rtt in rtts.iteritems() if rtt.rtt]
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
	def get_fake_IAT(self):
		IAT = [self.epsilon]
		for x in range(1, self.num_packets):
			t = self.timestamps[x]-self.timestamps[x-1]
			if t>0:
				IAT.append(t)
			else:
				IAT.append(self.epsilon)
		return IAT
        
	# Get the IAT array of a trace
	def get_IAT(self):
                sign = lambda x: x and (1, -1)[x < 0]
		IAT = []
		for x in range(0, self.num_packets):
                        y = x + 1
                        while y < self.num_packets and \
                              sign(self.packetsizes[x]) != sign(self.packetsizes[y]):
                                y = y + 1
			if y < self.num_packets and \
                           self.timestamps[y] != self.timestamps[x]:
				IAT.append(self.timestamps[y] - self.timestamps[x])
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

class RTT:
        """ Represent the round trip time (RTT) of a TCP connection. """
        def __init__(self, time, seq, src, dst):
                self.syn_time = time
                self.seq = seq
                self.ack = None
                self.rtt = None
                self.src = src
                self.dst = dst

        def __repr__(self):
                rstr = "<RTT syn_time:{} seq:{} ack:{} rtt: {} src:{} dst:{}>"
                return rstr.format(self.syn_time, self.seq, self.ack, self.rtt, self.src, self.dst)

        def set_ack(self, ack):
                """
                Sets the ACK number from a SYN/ACK packet.
                @param ack: The ACK number from a SYN/ACK packet.
                """
                self.ack = ack

        def set_rtt(self, t):
                """
                Calculates the initial RTT of a TCP connection.
                @param t: The time an ACK to a SYN/ACK is sent.
                """
                self.rtt = t - self.syn_time

        def get_rtt(self):
                """
                Returns the RTT. Will be None if the RTT has not been set with set_rtt.
                """
                return self.rtt
