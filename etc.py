import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sn

speeds = [*reversed([0.1, 0.2, 0.4, 0.8, 1, 1.25, 2.5, 5, 10])]
acc = [0.793, 0.7945, 0.7866, 0.7601, 0.88775, 0.6819, 0.3792, 0.10894, 0.0490]

#plt.scatter(speeds, acc)
#plt.savefig('plt.jpg')
with sn.plotting_context("notebook", font_scale=3):
    fig = plt.figure(figsize=(30,10), )
    spec = gridspec.GridSpec(nrows=1, ncols=1, hspace=0.1, wspace=.5,
                                figure=fig)
    ax = fig.add_subplot(spec[0])
    ax.axvline(1.0, color='red', linestyle='--', linewidth=5)

    #sn.lineplot(data=acc, x='adj_scale', y='perf', marker='o', markersize=10, 
    #        linewidth=5, ax=ax)

    ax.set_xscale('log')

    ax.set_ylim(-0.05,1.05)
    #plt.xticks([.1, .2, .4, .8, 1.0, 1.25, 2.5, 5.0, 10.], 
    #           [.1, .2, .4,"", 1.0, "", 2.5, 5.0, 10.])
    ax.set_xticks([.1,.2,.4,1.0,1.25, 2.5,5.,10.])
    ax.set_xticklabels([.1,.2,.4,1.0,1.25, 2.5,5.,10.], 
                        )
    ax.set_xlabel('Scale')
    ax.set_ylabel('Accuracy')
    ax.grid()

    ax.plot(speeds, acc, linewidth=5)
    ax.scatter(speeds, acc, s=100, marker='o')

    plt.savefig('plt.jpg')