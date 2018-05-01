import numpy as np
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix



class DataPreprocessing(object):
	def __init(self):
		pass

	def csr_info(self,mat, name="", non_empty=False):
		""" Print out info about this CSR matrix. If non_empy, 
		report number of non-empty rows and cols as well
		"""
		if non_empty:
			print("%s [nrows %d (%d non-empty), ncols %d (%d non-empty), nnz %d]" % (
				name, mat.shape[0], 
				sum(1 if mat.indptr[i+1] > mat.indptr[i] else 0 
				for i in range(mat.shape[0])), 
				mat.shape[1], len(np.unique(mat.indices)), 
				len(mat.data)))
		else:
				print( "%s [nrows %d, ncols %d, nnz %d]" % (name, 
				mat.shape[0], mat.shape[1], len(mat.data)) )

	def csr_idf(self,mat, copy=False, **kargs):
		""" Scale a CSR matrix by idf. 
		Returns scaling factors as dict. If copy is True, 
		returns scaled matrix and scaling factors.
		"""
		if copy is True:
				mat = mat.copy()
		nrows = mat.shape[0]
		nnz = mat.nnz
		ind, val, ptr = mat.indices, mat.data, mat.indptr

		df = defaultdict(int)
		for i in ind:
			df[i] += 1

		for k,v in df.items():
			df[k] = np.log(nrows / float(v)) 

		for i in range(0, nnz):
			val[i] *= df[ind[i]]

		return df if copy is False else mat


	def csr_build(self, docs, indices, values):
		nrows = len(docs)
		idx = {}
		# tid = 0
		nnz = 0
		max_ind = []
		for d in indices:
			nnz += len(d)
			max_ind.append(max(d))
			'''
			for w in d:
				if w not in idx:
					idx[w] = tid
					tid += 1
			'''

		ind = np.zeros(nnz, dtype=np.int)
		val = np.zeros(nnz, dtype=np.int)
		ptr = np.zeros(nrows+1, dtype=np.int)
		i = 0  # document ID / row counter
		n = 0  # non-zero counter

		for idxs,count in zip(indices,values):
			l = len(idxs)
			for j in range(l):
				ind[j + n] = idxs[j]
				val[j + n] = count[j]
			ptr[i+1] = ptr[i] + l
			n += l
			i += 1
		ncols = max(ind) + 1
		print ptr
		print ind
		print val

		mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
		return mat

	def csr_l2normalize(self, mat, copy=False, **kargs):
		r""" Normalize the rows of a CSR matrix by their L-2 norm. 
		If copy is True, returns a copy of the normalized matrix.
		"""
		if copy is True:
			mat = mat.copy()
		nrows = mat.shape[0]
		nnz = mat.nnz
		ind, val, ptr = mat.indices, mat.data, mat.indptr
		# normalize
		for i in range(nrows):
			rsum = 0.0    
			for j in range(ptr[i], ptr[i+1]):
				rsum += val[j]**2
			if rsum == 0.0:
				continue  # do not normalize empty rows
			rsum = 1.0/np.sqrt(rsum)
			for j in range(ptr[i], ptr[i+1]):
				val[j] *= rsum

		if copy is True:
			return mat
