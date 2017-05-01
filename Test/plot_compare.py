
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt

def plot_compare(X_compare,S_compare,S_ica):

    pca = PCA(n_components=3)
    S_pca = pca.fit_transform(X_compare)

    fa = FactorAnalysis(n_components=3)
    S_fa= fa.fit_transform(X_compare)

    models = [X_compare, S_compare,S_ica, S_pca, S_fa]
    names = ['Observations (mixed signal)',
             'True Sources',
             'FastICA recovered IC signals',
             'PCA recovered IC signals',
            'Factor Analysis recovered IC signals']
    colors = ['red', 'steelblue', 'orange']


    plt.figure(figsize=(10,6))
    for ii, (model, name) in enumerate(zip(models, names), 1):  # enumerate starts from 1
        plt.subplot(5, 1, ii)
        plt.title(name)
        for sig, color in zip(model.T, colors):
            plt.plot(sig, color=color)
            plt.xticks([])
            plt.yticks([])
    plt.subplots_adjust(0.09, 0.09, 0.94, 0.94, 0.5, 1)