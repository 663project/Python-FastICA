
import numpy as np
import matplotlib.pyplot as plt

def plot_finance(X_finance,ica_finance):
    models = [np.array(X_finance), ica_finance['S'],ica_finance['S'] @ ica_finance['A']]
    names = ['Observations (mixed signals)',
             'ICA recovered signals',
            'Reconstructed signals']
    colors = ['red', 'steelblue', 'orange','black','blue','yellow']

    plt.figure(figsize=(10,6))
    for ii, (model, name) in enumerate(zip(models, names), 1):  # enumerate starts from 1
        plt.subplot(4, 1, ii)
        plt.title(name)
        for sig, color in zip(model.T, colors):
            plt.plot(sig, color=color)
            plt.xticks([])
            plt.yticks([])

    plt.subplots_adjust(0.09, 0.09, 0.94, 0.94, 0.5, 1)

    X_re = ica_finance['S'] @ ica_finance['A']
    model100 = models = [np.array(X_finance)[:100,],X_re[:100,],ica_finance['S'][:100,0],ica_finance['S'][:100,1],ica_finance['S'][:100,2],ica_finance['S'][:100,3],ica_finance['S'][:100,4],ica_finance['S'][:100,5]]
    names = ['Observations (recent 100 trading days)',
            'Reconstruct signals (recent 100 trading days)',
            'IC 1 (recent 100 trading days)',
            'IC 2 (recent 100 trading days)',
            'IC 3 (recent 100 trading days)',
            'IC 4 (recent 100 trading days)',
            'IC 5 (recent 100 trading days)',
            'IC 6 (recent 100 trading days)']
    colors = ['red', 'steelblue', 'orange','black','blue','yellow']

    plt.figure(figsize=(10,6))
    for ii, (model, name) in enumerate(zip(model100, names), 1):  # enumerate starts from 1
        plt.subplot(4, 2, ii)
        plt.title(name)
        if ii <3:
            for sig, color in zip(model.T, colors):
                plt.plot(sig, color=color)
                plt.xticks([])
                plt.yticks([])
        else:
            plt.plot(model, color='green')
            plt.xticks([])
            plt.yticks([])

    plt.subplots_adjust(0.09, 0.09, 0.94, 0.94, 0.5, 1)