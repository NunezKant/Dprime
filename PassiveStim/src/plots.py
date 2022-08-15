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