import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
from itertools import  product, combinations

def comparisonplot(data, figsize = (10,10), x="ID", y="Margin", hue="Layer", pairs=None, test=None, correction=None):

    def overall_margins_fromdf(data,x,y,hue):
        overall_margins = []
        for id in data[x].unique():
            for layer in data[hue].unique():
                selection = (data[x]==id) & (data[hue]==layer)
                overall_margin = data[selection][y].mean()
                overall_margins.append(overall_margin)
        return np.array(overall_margins)

    def get_text_xcordinates(data,x):
        xcoord = []
        n_ids = len(data[x].unique())
        for id in range(n_ids):
            x1 = id-0.2
            x2 = id+0.2
            xcoord.append(x1)
            xcoord.append(x2)
        return xcoord

    def annotate_means(ax, data):
        xcoord = get_text_xcordinates(data,x)
        means = overall_margins_fromdf(data,x,y,hue)
        for xc, mean in zip(xcoord,means):
            ax.annotate(str(np.round(mean,2)),xy=(xc,mean), ha="left", fontsize=15)

    def stat_comparisons(ax,data,x,y,hue,test,pairs=pairs):
        list_1 = data[x].unique()
        list_2 = data[hue].unique()
        if pairs is None:
            pairs = []
            products = list(product(list_1,list_2))
            pairs = list(combinations(products,2))
        annotator = Annotator(ax,pairs,x=x,y=y,hue=hue,data=data)
        annotator.configure(test=test,verbose=False,comparisons_correction=correction,line_offset_to_group=.9).apply_and_annotate()



    plt.figure(figsize=figsize)
    ax = sns.violinplot(x=x,y=y,hue=hue,data=data);
    plt.setp(ax.collections, alpha=.3)
    ax = sns.stripplot(x=x,y=y,hue=hue,data=data, dodge=True);
    annotate_means(ax, data)
    if test is not None:
        stat_comparisons(ax,data,x,y,hue,test,pairs)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[:2],labels[:2],title='Layer')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    sns.despine()

def neurondensity(Sessions,ovrl_gen_neu_per_session,figsize=(19,8)):
    from matplotlib.offsetbox import AnchoredText
    layers = 2
    sessions = len(Sessions)
    f, axs = plt.subplots(layers, sessions, figsize=figsize)
    for layer in range(layers):
        for ix, sess in enumerate(Sessions):
            mask = ovrl_gen_neu_per_session[ix][layer]
            axs[layer,ix].scatter(sess.xpos,-sess.ypos, s=0.05, alpha=0.5)
            sns.kdeplot(x=sess.xpos[mask], y=-sess.ypos[mask], fill=True, cmap="rocket", levels=100, alpha=0.3, weights=np.tile(1/len(sess.xpos),len(sess.xpos[mask])), ax = axs[layer,ix])
            axs[layer,ix].set_xticks([])
            axs[layer,ix].set_yticks([])
            anchored_text = AnchoredText(f"n = {len(mask)}", loc=4)
            axs[layer,ix].add_artist(anchored_text)
            if layer == 0:
                axs[layer,ix].set_title(f"{sess.name}")
    pad = 5 
    for ax, row in zip(axs[:,0], ["layer 1", "layer 2/3"]):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size=14, ha='right', va='center')