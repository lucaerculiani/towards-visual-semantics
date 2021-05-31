#import matplotlib
#matplotlib.use("Agg")

import argparse
import matplotlib.pyplot as plt
import numpy as np
import recsiam.openworld as ow

plt.rcParams.update({'font.size': 15})

_TITLE = True

def main(cmdline):

    sup_s, sup_e, sup_i = np.load(cmdline.supervised)
    act_s, act_e, act_i = np.load(cmdline.active)
    if not cmdline.only_incremental_metrics:

        plot_supervision(sup_s, act_s, cmdline.output_file, discard=cmdline.discard_first)
        plot_session_acc(sup_s, act_s, cmdline.output_file, discard=cmdline.discard_first)
        if act_e["neigh"].size > 0 and sup_e["neigh"].size > 0:
            plot_eval_prec_rec(sup_s, sup_e, act_s, act_e, cmdline.output_file, discard=cmdline.discard_first)
        else:
            print("skipping evaluation plot due to missing data")


        if "n_ask" in sup_s and "n_ask" in act_s:
            plot_all_supervision(sup_s, act_s, cmdline.output_file, discard=cmdline.discard_first)
        else:
            print("skipping complete supervision plot due to missing data")

        if cmdline.clustering:
            if "cc" in sup_s and "cc" in act_s:
                plot_rand_index(sup_s, act_s, cmdline.output_file, discard=cmdline.discard_first)
                plot_adjusted_mutual_information(sup_s, act_s, cmdline.output_file, discard=cmdline.discard_first)
            else:
                print("skipping conected components plot due to missing data")

    print_incremental_eval_stats(sup_s, sup_i, act_s, act_i)






def print_incremental_eval_stats(sup_s, sup_i, act_s, act_i):
    sup_cl_met = (ow.eval_ari(sup_i), ow.eval_ami(sup_i))
    sup_cl_met_m = tuple(elem.mean() for elem in sup_cl_met)
    print("FULLower ari {}\tnmi: {}".format(*sup_cl_met_m))
    act_cl_met = (ow.eval_ari(act_i).mean(), ow.eval_ami(act_i).mean())
    act_cl_met_m = tuple(elem.mean() for elem in act_cl_met)
    print("follower ari {}\tnmi: {}".format(*act_cl_met_m))

    sup_unk, sup_kn, sup_acc = ow.incremental_evaluation_accuracy(sup_s, sup_i)
    p_str = "{} unk rec {}, known rec: {}, accuracy{}"
    print(p_str.format("FULLower", sup_unk.mean(), sup_kn.mean(), sup_acc.mean()))

    act_unk, act_kn, act_acc = ow.incremental_evaluation_accuracy(act_s, act_i)
    print(p_str.format("follower", act_unk.mean(), act_kn.mean(), act_acc.mean()))


    print("###LATEX###")
    sup_lcol = np.around([sup_s["n_ask"].sum(axis=1).mean(), sup_acc.mean(), *sup_cl_met_m], decimals=2).astype(str)
    sup_std = np.around([0, sup_s["n_ask"].sum(axis=1).std(), sup_unk.std(), sup_kn.std(), *(elem.std() for elem in sup_cl_met)], decimals=3).astype(str)
    act_lcol = np.around([act_s["n_ask"].sum(axis=1).mean(), act_acc.mean(), *act_cl_met_m], decimals=2).astype(str)
    act_std = np.around([0, act_s["n_ask"].sum(axis=1).std(), act_unk.std(), act_kn.std(), *(elem.std() for elem in act_cl_met)], decimals=3).astype(str)

#    print(" & ".join(np.char.add(np.char.add(sup_lcol, " +- "), sup_std)) + " \\\\")
#    print(" & ".join(np.char.add(np.char.add(act_lcol, " +- "), act_std)) + " \\\\")
#    print(" & ".join(sup_lcol) + " \\\\")
#    print(" & ".join(act_lcol) + " \\\\")
    print(" & ".join(np.concatenate((sup_lcol, act_lcol))) + " \\\\")



def plot_all_supervision(sup_s, act_s, output_file, discard=1):
    ara = np.arange(discard, sup_s["pred"].shape[1] + 1)
    plt.clf()
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    ax.grid()
    if _TITLE:
        ax.set_title("Test results")
    sup_ask = sup_s["n_ask"]
    act_ask = act_s["n_ask"]

    ax.plot(ara, act_ask.mean(axis=0)[discard -1 :], 'b-', label="Follower")
    ax.plot(ara, sup_ask.mean(axis=0)[discard -1 :], 'r-', label="FULLower")
    print_str = "{} average number of queries {}"
    print(print_str.format("FULLower", sup_ask.sum(axis=1).mean()))
    print(print_str.format("follower", act_ask.sum(axis=1).mean()))
#    ax.legend(loc=9)
#    ax.set_ylim(-0.05,1.05)
    fig.savefig(output_file + "n_ask.png")



def plot_rand_index(sup_s, act_s, output_file, discard=1):
    ara = np.arange(discard, sup_s["pred"].shape[1] + 1)
    plt.clf()
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    ax.grid()
    if _TITLE:
        ax.set_title("Test results")
    sup_ari = ow.session_ari(sup_s)
    act_ari = ow.session_ari(act_s)

    ax.plot(ara, act_ari.mean(axis=0)[discard -1 :], 'b-', label="Follower")
    ax.plot(ara, sup_ari.mean(axis=0)[discard -1 :], 'r-', label="FULLower")
#    ax.legend(loc=9)
    ax.set_ylim(-0.05,1.05)
    fig.savefig(output_file + "ari.png")

def plot_adjusted_mutual_information(sup_s, act_s, output_file, discard=1):
    ara = np.arange(discard, sup_s["pred"].shape[1] + 1)
    plt.clf()
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    ax.grid()
    ax.set_title("Test results")
    sup_ari = ow.session_ami(sup_s)
    act_ari = ow.session_ami(act_s)

    ax.plot(ara, act_ari.mean(axis=0)[discard -1 :], 'b-', label="Follower")
    ax.plot(ara, sup_ari.mean(axis=0)[discard -1 :], 'r-', label="FULLower")
#    ax.legend(loc=9)
    ax.set_ylim(-0.05,1.05)
    fig.savefig(output_file + "ami.png")


def plot_eval_prec_rec(sup_s, sup_e, act_s, act_e, output_file, discard=1):
    ara = np.arange(discard, sup_s["pred"].shape[1] + 1)
    plt.clf()
    fig = plt.figure(figsize=(6,3.5))
    ax = fig.add_subplot(111)
    ax.grid()
    if _TITLE:
        ax.set_title("Test results")
    ax.set_ylabel("fraction of recognized objects")
    sup_prec, sup_rec = ow.evaluation_seen_unseen_acc(sup_s, sup_e)
    act_prec, act_rec = ow.evaluation_seen_unseen_acc(act_s, act_e)

    ax.plot(ara, act_prec.mean(axis=0)[discard -1 :], 'b-', label="Follower seen")
    ax.plot(ara, sup_prec.mean(axis=0)[discard -1 :], 'r-', label="FULLower seen")
    ax.plot(ara, act_rec.mean(axis=0)[discard -1 :], 'm-', label="Follower unseen")
    ax.plot(ara, sup_rec.mean(axis=0)[discard -1 :], "-.", color="black", label="FULLower unseen")
    ax.legend(loc=9)
    ax.set_ylim(-0.05,1.05)
#    print("sup final prec:\t{}".format(sup_prec[-1]))
#    print("Follower final prec:\t{}".format(act_prec[-1]))

    plt.subplots_adjust(top=0.99, bottom=0.08)
    fig.savefig(output_file + "precrec.png")
    

def plot_session_acc(sup_s, act_s, output_file, discard=1):
    ara = np.arange(discard, sup_s["pred"].shape[1] + 1)
    plt.clf()
    fig = plt.figure(figsize=(6,3.5))
    ax = fig.add_subplot(111)
    ax.grid()
    if _TITLE:
        ax.set_title("\"Instantaneous\" accuracy")
    ax.set_ylabel("\"Instantaneous\" accuracy")
    act_acc = ow.session_accuracy(act_s, by_step=True)[discard -1 :]
    sup_acc = ow.session_accuracy(sup_s, by_step=True)[discard -1 :]
    print("FULLower mean acc {}".format(sup_acc.mean().round(3)))
    print("follower mean acc {}".format(act_acc.mean().round(3)))
    if "n_embed" in sup_s:
        print("fULLower mean n_embed {}".format(sup_s["n_embed"].mean()))
    if "n_embed" in act_s:
        print("follower mean n_embed {}".format(act_s["n_embed"].mean()))

    ax.plot(ara, act_acc, 'b-', label="Follower")
    ax.plot(ara, sup_acc, 'r-', label="FULLower")
    ax.legend(loc=(0.6,0.7))
    ax.set_ylim(-0.05,1.05)

    plt.subplots_adjust(top=0.99, bottom=0.08)
    fig.savefig(output_file + "total_acc.png")
    #plt.show()


def plot_supervision(sup_s, act_s, output_file, discard=1):
    ara = np.arange(discard, sup_s["pred"].shape[1])
    plt.clf()
    fig = plt.figure(figsize=(6,3.5))
    ax = fig.add_subplot(111)
    ax.grid()
    if _TITLE:
        ax.set_title("Supervision")
    ax.set_ylabel("queries")
    ax.set_ylim([-0.05, 1.05])
    ax2 = ax.twinx()
    ax2.plot(np.arange(sup_s["pred"].shape[1]) , (1 - new_obj_frac(sup_s)) * np.unique(sup_s["class"][0]).size, 'g-')
    ax2.set_ylabel('unseen objects', color='g')
    ax2.tick_params("y", colors='g')
    ax.plot(ara, sup_prob(act_s)[discard:], 'b-', label="Follower")
    ax.plot(ara, sup_prob(sup_s)[discard:], 'r-', label="FULLower")
    ax.legend(loc=(0.6,0.7))

    plt.subplots_adjust(top=0.99, bottom=0.08)
    fig.savefig(output_file + "sup_prob.png")
    #plt.show()


def gt_overtime(gt, sup):
    t_gt = (gt.astype(np.bool) & sup).cumsum(axis=1)

    return t_gt.mean(axis=0)

def new_obj_frac(s_d):
    no = np.concatenate([ow.new_obj_in_seq(s) for s in s_d["class"]])
    no = no.cumsum(axis=1)
    no = no / no[:, -1, None]
    return no.mean (axis=0)

def correct_uncorrect(data):
    
    correnct_unk = np.logical_not(data[:,0, :] | data[:,1,:])

    correct_known = data[:,0,:] & data[:,1,:] & data[:,2,:]

    correct = correnct_unk | correct_known

    acc = (correct.cumsum(axis=1) / np.arange(1, correct.shape[1] + 1)).mean(axis=0)

    assert not (correnct_unk & correct_known).any()

    return acc

def correct_uncorrect_prob(data):
    
    correnct_unk = np.logical_not(data[:,0, :] | data[:,1,:])

    correct_known = data[:,0,:] & data[:,1,:] & data[:,2,:]

    correct = correnct_unk | correct_known

    acc = correct.mean(axis=0)

    assert not (correnct_unk & correct_known).any()

    return acc

def correct_unk(data):
    
    correnct_unk = np.logical_not(data[:,0, :] | data[:,1,:])


    correct = correnct_unk 

    acc = (correct.cumsum(axis=1) / np.arange(1, correct.shape[1] + 1)).mean(axis=0)


    return acc

def correct_known(data):
    

    correct = data[:,0,:] & data[:,1,:] & data[:,2,:]


    acc = (correct.cumsum(axis=1) / np.arange(1, correct.shape[1] + 1)).mean(axis=0)


    return acc

def sup_prob(res_s):
    s = res_s["ask"]
    return s.mean(axis=0)

    

def seen_unseen(data):
    
    p = data[:,0,:]
    gt = data[:,1,:]


    correct = p == gt

    acc = (correct.cumsum(axis=1) / np.arange(1, p.shape[1] + 1)).mean(axis=0)
    
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("supervised", type=str,
                    help="a folder or a comma separated list of files containing the data to load")
    parser.add_argument("active", type=str,
                    help="a folder or a comma separated list of files containing the data to load")
    parser.add_argument("-o", "--output-file", default='plot',type=str,
		       help="output file name")
    parser.add_argument("--discard-first", default=10,type=int,
		       help="discard first N objects")
    parser.add_argument("-c", "--clustering", action="store_true",
		       help="print metrics on clutering")
    parser.add_argument("-m", "--only-incremental-metrics", action="store_true",
		       help="print only incremental metrics")
    parser.add_argument("--no-title", action="store_false",
		       help="print only incremental metrics")
    args = parser.parse_args()

    _TITLE = args.no_title
    result = main(args)

