from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os 
class GradientVisualizer:
    def __init__(self, anchor=0):
        self.TSNE = TSNE(n_components=2, init='random', random_state=0, perplexity=5)
        self.anchor = anchor
        self.tensor = []
        self.colors = []
        self.labels = []

    def add(self, tensor, color, label):
        self.tensor.append(tensor.numpy())
        self.colors.append(color)
        self.labels.append(label)

    def plot(self, k):
        self.TSNE.fit_transform(np.array(self.tensor))
        X = self.TSNE.embedding_[:, 0]
        Y = self.TSNE.embedding_[:, 1]

        if self.anchor is not None:
            X = X - X[0]
            Y = Y - Y[0]

        assert(len(X) == len(self.colors))
        for i in range(len(X)):
            plt.scatter(X[i], Y[i], c=self.colors[i], label=self.labels[i])

        plt.title('Gradient TSNE')
        plt.xlabel('X')
        plt.ylabel('Y')
        save_dir='plot/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig('plot/{}.png'.format(k))
        plt.clf()

    def reset(self):
        self.tensor = []
        self.colors = []
        self.labels = []