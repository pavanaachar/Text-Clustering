import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_distances

class DBSCAN():
	def print_point_counts(self):
		core_pt_count = len(self.core_points)
		border_pt_count = len(self.border_points)
		noise_pt_count = len(self.noise_points)

		print "core_Points:", core_pt_count
		print "border_Points:", border_pt_count
		print "noise_Points:", noise_pt_count
		print "-------------"
		print "Total:", core_pt_count + border_pt_count + noise_pt_count

	def calculate_distances(self):
		#self.dist = cdist(self.data, self.data, 'euclidean')
		self.dist = cosine_distances(self.data, self.data)

	def __init__(self, data, epsilon, r, dist=None):
		self.data = data # numpy ndarray
		self.epsilon = epsilon
		self.density = r

		# Classified points
		self.core_points = []
		self.border_points = []
		self.noise_points = []

		# Graph
		self.graph = nx.Graph()
		self.labels = []

		# Distance matrix
		self.dist = dist

		# Sorted distances
		self.sorted_dist = dist.argsort()

	def points_closer_than_eps(self, point, min_pts=None):
		points = []
		for i in self.sorted_dist[point]:
			if self.dist[point][i] > self.epsilon:
				break
			else:
				points.append(i)
				if min_pts != None and len(points) >= min_pts:
					return points
		return points

	def classify_points(self):
		for idx in range(len(self.data)):
			# points = self.points_closer_than_eps(idx, self.density + 1) # +1 including self
			points = self.points_closer_than_eps(idx)
			points.remove(idx)

			if len(points) >= self.density:
				self.core_points.append(idx)

		for idx in range(len(self.data)):
			if idx in self.core_points:
				continue

			# points = self.points_closer_than_eps(idx, self.density + 1) # +1 including self
			points = self.points_closer_than_eps(idx)
			points.remove(idx)

			if len(points) > 0:
				core_found = False
				for p in points:
					if p in self.core_points:
						self.border_points.append(idx)
						core_found = True
						break
				if not core_found:
					self.noise_points.append(idx)
			else:
				self.noise_points.append(idx)

		self.print_point_counts()

	def connect_core_points(self):
		count = 0
		core_points = list(self.core_points)

		while len(core_points) > 0:
			pt = core_points.pop()
			self.graph.add_node(pt)
			count += 1

			points = self.points_closer_than_eps(pt)
			points.remove(pt)

			for p in points:
				if p in self.core_points:
					self.graph.add_edge(pt, p)
					if p in core_points:
						core_points.remove(p)
						count += 1
		
		print("Connected core points: " + str(count))

	def connect_border_points(self):
		clusters = []
		for conn_comp in nx.connected_components(self.graph):
			clusters.append(conn_comp)

		for bp in self.border_points:
			cluster_counts = [0] * len(clusters)

			points = self.points_closer_than_eps(bp)
			points.remove(bp)

			for p in points:
				for cluster_idx in range(len(clusters)):
					if p in clusters[cluster_idx]:
						cluster_counts[cluster_idx] += 1
						break

			assert max(cluster_counts) != 0
			target_cluster_idx = cluster_counts.index(max(cluster_counts))
			# print(clusters[target_cluster_idx].pop())
			connect_to = clusters[target_cluster_idx].pop()
			# print bp, ",", connect_to
			self.graph.add_edge(connect_to, bp)
			clusters[target_cluster_idx].add(connect_to)

		print "Number of graph nodes after assigning border points:", nx.number_of_nodes(self.graph)

	def assign_labels(self):
		totalCount = 0
		count = 1

		self.labels = [None for i in range(len(self.core_points) + len(self.border_points) + len(self.noise_points))]
		'''
		clusters = nx.connected_components(self.graph)
		for cluster in clusters:
			print cluster
		'''

		for cluster in nx.connected_components(self.graph):
			for node in cluster:
				# print node
				self.labels[node] = count
				totalCount += 1
			count += 1

		for noise in self.noise_points:
			self.labels[noise] = count
			totalCount += 1

		print("totalCount:", totalCount)
		return

	def get_labels(self):
		labels = []
		for point in range(len(self.core_points) + len(self.border_points) + len(self.noise_points)):
			labels.append(self.labels[point])
		return labels


