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
        # comprovar que no siguin les mateixes!
    return selected_features


with open('/home/jpons/Desktop/istrainFalse_noDrop_batch1_1514562403.pkl', 'rb') as f:
    x_false, y_false, config = pickle.load(f)
X_false = select_cnn_feature_layers(x_false, [0,1,2,3,4])

with open('/home/jpons/Desktop/istrainTrue_noDrop_batch100_1514559642.pkl', 'rb') as f:
    x_true, y_true = pickle.load(f)
X_true = select_cnn_feature_layers(x_true, [0,1,2,3,4])

X_true_tsne = TSNE(n_components=2, verbose=1, perplexity=200, n_iter=1000, learning_rate=100).fit_transform(X_true)
X_false_tsne = TSNE(n_components=2, verbose=1, perplexity=200, n_iter=1000, learning_rate=100).fit_transform(X_false)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X_true_tsne[:, 0], X_true_tsne[:, 1], c=y_true, cmap=plt.cm.tab10)
plt.subplot(122)
plt.scatter(X_false_tsne[:, 0], X_false_tsne[:, 1], c=y_false, cmap=plt.cm.tab10)
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
