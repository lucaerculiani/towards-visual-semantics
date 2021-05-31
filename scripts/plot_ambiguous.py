#import matplotlib
#matplotlib.use("Agg")

import argparse
import matplotlib.pyplot as plt
import numpy as np
import recsiam.openworld as ow
from pathlib import Path
from tabulate import tabulate
import lz4

plt.rcParams.update({'font.size': 15})

_TITLE = True
_COLORS = "brgcmyk"
_STYLES = [c+"-" for c in _COLORS]



def smart_load(path):
    if path.endswith(".lz4"):
        with lz4.frame.open(str(path), mode="rb") as f:
            loaded = np.load(f)[0]
    else:
        loaded = np.load(path)

    return loaded

def main(cmdline):

    data = np.array([smart_load(f) for f in cmdline.i_files])

    if cmdline.labels is None:
        labels = ["data_"+str(i) for i in range(1, len(data)+1)]
    else:
        labels = cmdline.labels.split(",")

    assert len(labels) == len(data)


    basenames = ["inst_e_acc", "sup", "gdn_acc"]
    base_path = Path(cmdline.output_file)
    if base_path.is_dir():
        outputs = [base_path / b for b in basenames]
    else:
        outputs = [str(base_path) + b for b in basenames]

    tables = [
#            np.vectorize(compute_conf_mat, signature="(),(),()->(n,m,k)")(data[:, 0, 0], data[:, 1, 0], data[:, 1, 1])[:, -1, ...]
            ]

    yss = [
        np.vectorize(compute_inst_acc, signature="()->(n)")(data[:, 0]),
#        np.vectorize(compute_total_supervision, signature="()->(n)")(data[:, 1, 0]),
#        np.vectorize(compute_gen_diff_perf, signature="(),(),()->(n)")(data[:, 0, 0], data[:, 1, 0], data[:, 1, 1])
            ]

    discard = [10]
    ylims = [(0.1, 0.3)]
    ylabs = ["accuracy"]
    xlabs = ["iteration"] 

    for t in tables:
        print(tabulate(list(zip(labels, t))))

    for ys, o, d, ylim, yl, xl in zip(yss, outputs, discard, ylims, ylabs, xlabs):
        plot_generic(ys, None, labels, o, discard=d, ylim=ylim, ylab=yl, xlab=xl)

#    for y in yss:
#        print(tabulate(list(zip(labels, y[:, -1]))))


#    for i in range(data.shape[0]):
#        plot_scatter(*compute_scatter(data[i, 0, 0], data[i, 1, 0], data[i, 1, 1]),
#                     base_path / (labels[i] + "scatter.png"))

def plot_scatter(y, x, output_file):
    plt.clf()
    fig = plt.figure(figsize=(6, 4))

    ax = fig.add_subplot(111)
    ax.grid()
#    ax.set_xlim(-0.05)
    ax.set_ylim(0.0, 0.5)
#    if _TITLE:
#        ax.set_title("Test results")

    perc = np.percentile(x, [25,50,75,100])
    last = min(x)
    ax.axhline(y=y.mean(), color="g", linestyle="--")
    ax.axvline(x=last, color="k")
    for p in perc:
        ax.axvline(x=p, color="k")
        yperc = [y[(x >= last) & (x < p)].mean()]
        for yp in yperc:
            ax.plot([last,p], [yp,yp], "r-")

        last = p
    ax.scatter(x, y, s=20)
    ax.legend()
    fig.savefig(str(output_file) + ".png")




def plot_generic(ys, x, labels, output_file, discard=0, **kwargs):

    if x is None:
        x = np.arange(len(ys[0]))
    plt.clf()
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.grid()
    if "ylim" in kwargs and kwargs["ylim"] is not None:
        ax.set_ylim(*kwargs["ylim"])
    if "ylab" in kwargs and kwargs["ylab"] is not None:
        ax.set_ylabel(kwargs["ylab"])
    if "xlab" in kwargs and kwargs["xlab"] is not None:
        ax.set_xlabel(kwargs["xlab"])
#    ax.set_xlim(-5)
#    if _TITLE:
#        ax.set_title("Test results")

    for y, l, s in zip(ys, labels, _STYLES):
        ax.plot(x[discard:], y[discard:], s, label=l)
    ax.legend()
    fig.tight_layout()
    #plt.subplots_adjust(top=0.99, bottom=0.08)
    fig.savefig(str(output_file) + ".png")



def compute_scatter(session_d, amb_session_d, amb_eval_d):
    m = ow.genus_diff_novel_confusion_matrix(session_d, amb_session_d, amb_eval_d)

    y = (session_d["n_embed"] + amb_session_d["n_embed"]).squeeze()
#    y = amb_session_d["n_embed"]
    m = m
    x = (m[:, :, 0, 0] + m[:, :, 1, 1] + m[:, :, 2, 2]).mean(axis=1).squeeze()

    return x, y


def compute_inst_acc(session_d):

    acc = ow.session_accuracy(session_d, by_step=True)

    ret = acc.cumsum()

    w = 5
    ret[w:] = ret[w:] - ret[:-w]
    return np.concatenate((acc[:w-1], ret[w - 1:] / w))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("i_files", type=str, nargs='+',
                        help="list of file sto load")
    parser.add_argument("-o", "--output-file", default='plot', type=str,
                        help="output file name")
    parser.add_argument("--discard-first", default=10, type=int,
                        help="discard first N objects")
    parser.add_argument("-l", "--labels", default=None, type=str,
                        help="labels for files")
    args = parser.parse_args()

    result = main(args)

