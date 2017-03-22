import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
%matplotlib inline


def draw_box_plot(y_values, x_labels, x_label, y_label):
    plt.rcParams['figure.subplot.bottom'] = 0.23  # keep labels visible
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # make plot larger in notebook
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_opts={'cutoff_val':5, 'cutoff_type':'abs',
                'label_fontsize':'small',
                'label_rotation':0,
                'jitter_marker':'.',
                'jitter_marker_size':5,
                'bean_color':'#FF6F00',
                'bean_mean_color': 'g'
              }
    sm.graphics.beanplot(y_values, ax=ax, labels=x_labels, jitter=True, plot_opts=plot_opts)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    data['Age'] = data['Age'].fillna(np.nanmedian(data['Age']))
    labels = ["Survived", "Not Survived"]
    age = [data['Age'][data['Survived'] == i] for i in [0, 1]]
    draw_box_plot(age, labels, "Survived?", "Age")


def test_mosaic_plot(cat_target, cat_feature, data):
    from statsmodels.graphics.mosaicplot import mosaic
    mosaic(data, [cat_feature, cat_target])


def density(num_target, cat_feature, data):
    for val in data[cat_feature].unique():
        sns.kdeplot(data[num_target][data[cat_feature] == val], label=val)


def scatter(num_target, num_feature, data):
    sns.regplot(x=num_feature, y=num_target, data=data)


def box(cat_target, num_feature, data):
    f, ax = plt.subplots(figsize=(15, 10))
    vals = []
    names = []
    for val in data[cat_target].unique():
        vals.append(data[num_feature][data[cat_target] == val])
        names.append(val)
    sns.boxplot(vals=vals, names=names, ax=ax)


def main():
    print 'main'

if __name__ == "__main__":
    main()