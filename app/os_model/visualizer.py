import cv2
from PIL import Image

import os
import numpy as np

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#sns.set(style="whitegrid")


def vis_results_feature(features, rf_types, os_types, perturb_types, save=False, fname="im", root_dir="fig_attn",) :

    fig, axes = plt.subplots(nrows=3,ncols=1, figsize=(9,12));


    data_dict = {"feature_x": features[:, 0],
                 "feature_y": features[:, 1],
                 "rf_types": rf_types,
                 "os_types": os_types,
                 "perturb_types": perturb_types}

    df_data = pd.DataFrame(data_dict);

    sns.scatterplot(x="feature_x", y="feature_y", data=df_data,
                    hue="rf_types", s=25, alpha=1.0,  ax=axes[0])
    sns.scatterplot(x="feature_x", y="feature_y", data=df_data,
                    hue="perturb_types", s=25, alpha=1.0, ax=axes[1])
    sns.scatterplot(x="feature_x", y="feature_y", data=df_data.loc[df_data.perturb_types==0],
                    hue="os_types", s=25, alpha=1.0, ax=axes[2])

    for i in range(len(axes)) :
        axes[i].xaxis.set_ticklabels([])
        axes[i].yaxis.set_ticklabels([])
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        #axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        axes[i].legend(loc='upper left')

    #plt.show()
    plt.savefig(os.path.join(root_dir, fname))


def vis_results_shuffle(input1, input2, fixed_idxes=None, save=False, alpha=0.8, fname="im",
                     root_dir="fig_attn", nrows=1, ncols=1) :
    input1 = input1.astype(np.float32);
    input2 = input2.astype(np.float32);

    output_imgs = []

    W = 128;
    H = 128;

    w_per_one = W
    h_per_one = H * 3 + 60

    for idx, (i1,i2) in enumerate(zip(input1, input2)):
        i1[:, :5] = i1[:, :5] / 2.0;
        i1[:, 5:] = i1[:, 5:] - 3.0;
        i2[:, :5] = i2[:, :5] / 2.0;
        i2[:, 5:] = i2[:, 5:] - 3.0;

        i1 = np.repeat(np.expand_dims(i1, -1), 3, axis=2);
        i2 = np.repeat(np.expand_dims(i2, -1), 3, axis=2);

        i1 = (i1 * 255.0).astype(np.uint8)
        i2 = (i2 * 255.0).astype(np.uint8);

        if fixed_idxes is not None :
            f_i = (fixed_idxes[idx] * 255.0).astype(np.uint8);
            hmap_f = cv2.applyColorMap(f_i, cv2.COLORMAP_JET);


        i1 = cv2.resize(i1, dsize=(W, H),
                       interpolation=cv2.INTER_NEAREST_EXACT);
        i2 = cv2.resize(i2, dsize=(W, H),
                       interpolation=cv2.INTER_NEAREST_EXACT);
        hmap_f = cv2.resize(hmap_f, dsize=(W, H),
                            interpolation=cv2.INTER_LINEAR) * 0.6;

        olapped_img = hmap_f * alpha + (1 - alpha) * i1;
        olapped_img = olapped_img.astype(np.uint8)

        cat_img = np.zeros(shape=(w_per_one,
                                  h_per_one, 3), dtype=np.uint8);

        cat_img[:, :H] = i1
        cat_img[:, H + 20:H + 20 + H] = olapped_img
        cat_img[:, -H:] = i2

        output_imgs.append(cat_img)

    tot_img = np.zeros(shape=((w_per_one + 30) * nrows,
                              (h_per_one) * ncols, 3), dtype=np.uint8);

    for row in range(nrows):
        for col in range(ncols):
            img_idx = row * ncols + col;

            tot_img[img_idx * 30 + w_per_one * img_idx:
                    img_idx * 30 + w_per_one * img_idx + w_per_one,
            :] = output_imgs[img_idx]

    #cv2.imshow("test", tot_img)
    #cv2.waitKey(0)
    fname = os.path.join(root_dir, fname)

    # tot_img = Image.fromarray(tot_img);
    cv2.imwrite(fname, tot_img)


def vis_results_attn(data, attention, head_idx, save=False, alpha=0.8, fname="im",
                     root_dir="fig_attn", nrows=1, ncols=1) :
    data = data.astype(np.float32);
    attn_out = attention.astype(np.float32);

    _, a_heads, *_ = attn_out.shape;

    output_imgs = []

    W = 128;
    H = 128;

    w_per_one = W
    h_per_one = H * 3 + 60

    for d, c in zip(data, attn_out):
        d[:, :5] = d[:, :5] / 2.0;
        d[:, 5:] = d[:, 5:] - 3.0
        d = np.repeat(np.expand_dims(d, -1), 3, axis=2);
        c = c[head_idx]/c[head_idx].max();

        d = (d * 255.0).astype(np.uint8)
        c = (c * 255.0).astype(np.uint8);

        hmap_c = cv2.applyColorMap(c, cv2.COLORMAP_JET);

        d = cv2.resize(d, dsize=(W, H),
                       interpolation=cv2.INTER_NEAREST_EXACT);
        hmap_c = cv2.resize(hmap_c, dsize=(W, H),
                            interpolation=cv2.INTER_LINEAR) * 0.6;

        olapped_img = hmap_c * alpha + (1 - alpha) * d;
        olapped_img = olapped_img.astype(np.uint8)

        cat_img = np.zeros(shape=(w_per_one,
                                  h_per_one, 3), dtype=np.uint8);

        cat_img[:, :H] = d
        cat_img[:, H + 20:H + 20 + H] = hmap_c
        cat_img[:, -H:] = olapped_img

        output_imgs.append(cat_img)

    tot_img = np.zeros(shape=((w_per_one + 30) * nrows,
                              (h_per_one) * ncols, 3), dtype=np.uint8);

    for row in range(nrows):
        for col in range(ncols):
            img_idx = row * ncols + col;

            tot_img[img_idx * 30 + w_per_one * img_idx:
                    img_idx * 30 + w_per_one * img_idx + w_per_one,
            :] = output_imgs[img_idx]

    # cv2.imshow("test", tot_img)
    # cv2.waitKey(0)
    if not os.path.exists(root_dir) :
        os.makedirs(root_dir)
    fname = os.path.join(root_dir, fname)

    # tot_img = Image.fromarray(tot_img);
    cv2.imwrite(fname, tot_img)


def vis_results_cam(data, cam_out, labels, save=False, alpha=0.8, fname="im",
                root_dir="fig_cam", nrows=4, ncols=1):

    data = data.astype(np.float32);
    cam_out = cam_out.astype(np.float32);

    output_imgs = []

    W = 128;
    H = 128;

    w_per_one = W
    h_per_one = H*3 + 60

    for d, c, l in zip(data, cam_out, labels) :

        d[:,:5]= d[:,:5]/2.0;
        d[:,5:]= d[:,5:]-3.0
        d = np.repeat(np.expand_dims(d, -1), 3, axis=2);

        d = (d * 255.0).astype(np.uint8)
        c = (c * 255.0).astype(np.uint8);

        hmap_c = cv2.applyColorMap(c, cv2.COLORMAP_JET);

        d = cv2.resize(d, dsize=(W,H),
                                 interpolation= cv2.INTER_NEAREST_EXACT);
        hmap_c = cv2.resize(hmap_c, dsize=(W,H),
                                 interpolation= cv2.INTER_LINEAR) * 0.6;


        olapped_img = hmap_c*alpha + (1-alpha)*d;
        olapped_img = olapped_img.astype(np.uint8)

        cat_img = np.zeros(shape=(w_per_one,
                                  h_per_one, 3), dtype=np.uint8);

        cat_img[:,:H] = d
        cat_img[:,H+20:H+20 + H] = hmap_c
        cat_img[:,-H:] = olapped_img

        output_imgs.append(cat_img)

    tot_img = np.zeros(shape=((w_per_one + 30) * nrows,
                              (h_per_one) * ncols, 3), dtype=np.uint8);

    for row in range(nrows):
        for col in range(ncols) :
            img_idx = row*ncols + col;

            tot_img[img_idx*30 + w_per_one*img_idx :
                    img_idx*30 + w_per_one*img_idx + w_per_one,
                    :] = output_imgs[img_idx]

    #cv2.imshow("test", tot_img)
    #cv2.waitKey(0)
    fname = os.path.join(root_dir, fname)

    #tot_img = Image.fromarray(tot_img);
    cv2.imwrite(fname, tot_img)


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    #sns.set(style="white")
    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


