class Helper(object):
	def __init__(self, path):
		self.path = path
	
	def get_data(self):
		docs = []
		with open(self.path) as f:
			train_data = f.readlines()
		for doc in train_data:
			docs.append(doc.strip(' ').strip('\n').split(" "))
		return docs

	def separate_word_frequency(self, docs):
		indices = []
		values = []
		for doc in docs:
			word_ind = []
			word_count = []
			for i in range(0,len(doc),2):
				word_ind.append(doc[i])
			for i in range(1,len(doc),2):
				word_count.append(doc[i])
			indices.append(word_ind)
			values.append(word_count)
		return indices, values
