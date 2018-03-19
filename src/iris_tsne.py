from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
import numpy as np

def select_cnn_feature_layers(feature_maps, selected_features_list):
    selected_features = []
    for i in range(len(feature_maps)):
        tmp = np.array([])
        for j in selected_features_list:
            tmp = np.concatenate((tmp, np.squeeze(feature_maps[i][j])))
        selected_features.append(tmp)
    return selected_features


with open('/homedtic/jpons/elmarc/data/GTZAN/features/try.pkl', 'rb') as f:
    [x_train, y_train, id_train, x_val, y_val, id_val, x_test1, y_test1, id_test, config] = pickle.load(f)
X_test1 = select_cnn_feature_layers(x_test1, [0,1])

with open('/homedtic/jpons/elmarc/data/GTZAN/features/try.pkl', 'rb') as f:
    [x_train, y_train, id_train, x_val, y_val, id_val, x_test2, y_test2, id_test, config] = pickle.load(f)
X_test2 = select_cnn_feature_layers(x_test2, [0,1])

X1_tsne = TSNE(n_components=2, verbose=1, perplexity=200, n_iter=1000, learning_rate=100).fit_transform(X_test1)
X2_tsne = TSNE(n_components=2, verbose=1, perplexity=200, n_iter=1000, learning_rate=100).fit_transform(X_test2)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X1_tsne[:, 0], X1_tsne[:, 1], c=y_test1)#, cmap=plt.cm.tab10)
plt.subplot(122)
plt.scatter(X2_tsne[:, 0], X2_tsne[:, 1], c=y_test2)#, cmap=plt.cm.tab10)
plt.title('False: no norm, b1')
plt.show()


iris = load_iris()
X_tsne = TSNE(learning_rate=100).fit_transform(iris.data)
X_pca = PCA().fit_transform(iris.data)

print(iris.target)
print(iris.data)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=iris.target)
plt.subplot(122)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target)
plt.show()
