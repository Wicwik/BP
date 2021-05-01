import numpy as np
import sklearn.linear_model as linear_model

def find_feature_axis(z, y, method='linear', **kwargs_model):
	if method == 'linear':
		model = linear_model.LinearRegression(**kwargs_model)
		model.fit(z, y)
	elif method == 'tanh':
		def arctanh_clip(y):
			return np.arctanh(np.clip(y, np.tanh(0), np.tanh(3)))

		model = linear_model.LinearRegression(**kwargs_model)
		model.fit(z, arctanh_clip(y))

	return model.coef_.transpose()

def normalize_feature_axis(feature_slope):
	return feature_slope / np.linalg.norm(feature_slope, ord=2, axis=0, keepdims=True)

def orthogonalize_one_vector(vector, vector_base):
	return vector - np.dot(vector, vector_base) / np.dot(vector_base, vector_base) * vector_base

def orthogonalize_vectors(vectors):
	vectors_orthogonal = vectors + 0
	num_dimension, num_vector = vectors.shape

	for i in range(num_vector):
		for j in range(i):
			vectors_orthogonal[:, i] = orthogonalize_one_vector(vectors_orthogonal[:, i], vectors_orthogonal[:, j])

	return vectors_orthogonal

def disentangle_feature_axis(feature_axis_target, feature_axis_base, base_orthogonalized=False):
	if len(feature_axis_target.shape) == 0:
		single_vector_in = True
		feature_axis_target = feature_axis_target[:, None]
	else:
		single_vector_in = False

	if base_orthogonalized:
		feature_axis_base_orthononal = orthogonalize_vectors(feature_axis_base)
	else:
		feature_axis_base_orthononal = feature_axis_base

	feature_axis_decorrelated = feature_axis_target + 0
	num_dim, num_feature_0 = feature_axis_target.shape
	num_dim, num_feature_1 = feature_axis_base_orthononal.shape

	for i in range(num_feature_0):
		for j in range(num_feature_1):
			 feature_axis_decorrelated[:, i] = orthogonalize_one_vector(feature_axis_decorrelated[:, i], feature_axis_base_orthononal[:, j])

	if single_vector_in:
		return  feature_axis_decorrelated[:, 0]

	return feature_axis_decorrelated


def disentangle_feature_axis_by_idx(feature_axis, idx_base=None, idx_target=None, normalize=True):
	(num_dim, num_feature) = feature_axis.shape

	if idx_base is None or len(idx_base) == 0:
		feature_axis_disentangled = feature_axis
	else:
		if idx_target is None:
			idx_target = np.setdiff1d(np.arange(num_feature), idx_base)

		print(idx_target)

		feature_axis_target = feature_axis[:, idx_target] + 0
		feature_axis_base = feature_axis[:, idx_base] + 0
		feature_axis_base_orthogonalized = orthogonalize_vectors(feature_axis_base)
		feature_axis_target_orthogonalized = disentangle_feature_axis(feature_axis_target, feature_axis_base_orthogonalized, base_orthogonalized=True)

		feature_axis_disentangled = feature_axis + 0
		feature_axis_disentangled[:, idx_target] = feature_axis_target_orthogonalized
		feature_axis_disentangled[:, idx_base] = feature_axis_base_orthogonalized

	if normalize:
		return normalize_feature_axis(feature_axis_disentangled)

	return feature_axis_disentangled

def plot_feature_cos_sim(feature_direction, feature_name=None):
	import matplotlib.pyplot as plt
	from sklearn.metrics.pairwise import cosine_similarity

	len_z, len_y = feature_direction.shape
	if feature_name is None:
		feature_name = range(len_y)

	feature_cos_sim = cosine_similarity(feature_direction.transpose())
	c_lim_abs = np.max(np.abs(feature_cos_sim))

	plt.pcolormesh(np.arange(len_y+1), np.arange(len_y+1), feature_cos_sim, vmin=-c_lim_abs, vmax=+c_lim_abs, cmap='coolwarm')
	plt.gca().invert_yaxis()
	plt.colorbar()
	plt.xticks(np.arange(len_y) + 0.5, feature_name, fontsize='x-small', rotation='vertical')
	plt.yticks(np.arange(len_y) + 0.5, feature_name, fontsize='x-small')
	plt.show()

	return feature_cos_sim