"""
utility functions for the analysis of the data
"""
import os
from pyrsistent import s
from scipy import io, stats
import numpy as np
from suite2p.extraction import dcnv
from scipy import ndimage
from sklearn.metrics import accuracy_score


def add_exp(database, mname, expdate, blk):
    """
    Add an experiment to the database.

    Parameters
    ----------
    db : list
        List of experiments.
    mname : str
        Mouse name.
    expdate : str
        Experiment date.
    blk : str
        Block number.

    Returns
    -------
    db : list
        Updated List of experiments.
    """
    database.append({"mname": mname, "datexp": expdate, "blk": blk})
    return database  # Return the updated list


def load_exp(path, database, iexp):
    """
    Load an experiment from the database.

    Parameters
    ----------
    path : str
        Path to the experiment.
    db : list
        List of experiments.
    iexp : int
        Index of the experiment in the database.

    Returns
    -------
    Timeline: array
        Data of the experiment.
    ops: dict
        suite2p pipeline options
    """
    mname, datexp, blk = (
        database[iexp]["mname"],
        database[iexp]["expdate"],
        database[iexp]["block"],
    )
    root = os.path.join(path, mname, datexp, blk)
    fname = f"Timeline_{mname}_{datexp}_{blk}"
    fnamepath = os.path.join(root, fname)
    timeline = io.loadmat(fnamepath, squeeze_me=True)["Timeline"]

    ops = np.load(
        os.path.join(root, "suite2p", "plane0", "ops.npy"), allow_pickle=True
    ).item()
    return timeline, ops, root


def deconvolve(root, ops):
    """
    Correct the lags of the dcnv data.

    Parameters
    ----------
    root : str
        Path to the experiment.
    ops : dict
        suite2p pipeline options
    """

    # we initialize empty variables
    spks = np.zeros(
        (0, ops["nframes"]), np.float32
    )  # the neural data will be Nneurons by Nframes.
    stat = np.zeros((0,))  # these are the per-neuron stats returned by suite2p
    xpos, ypos = np.zeros((0,)), np.zeros((0,))  # these are the neurons' 2D coordinates

    # this is for channels / 2-plane mesoscope
    tlags = 0.25 + np.linspace(0.2, -0.8, ops["nplanes"] // 2 + 1)[:-1]
    tlags = np.hstack((tlags, tlags))

    # loop over planes and concatenate
    iplane = np.zeros((0,))

    th_low, th_high = 0.5, 1.1
    for n in range(ops["nplanes"]):
        ops = np.load(
            os.path.join(root, "suite2p", "plane%d" % n, "ops.npy"), allow_pickle=True
        ).item()

        # load and deconvolve
        iscell = np.load(os.path.join(root, "suite2p", "plane%d" % n, "iscell.npy"))[
            :, 1
        ]
        iscell = (iscell > th_low) * (iscell < th_high)

        stat0 = np.load(
            os.path.join(root, "suite2p", "plane%d" % n, "stat.npy"), allow_pickle=True
        )[iscell]
        ypos0 = np.array(
            [stat0[n]["med"][0] for n in range(len(stat0))]
        )  # notice the python list comprehension [X(n) for n in range(N)]
        xpos0 = np.array([stat0[n]["med"][1] for n in range(len(stat0))])

        ypos0 += ops["dy"]  # add the per plane offsets (dy,dx)
        xpos0 += ops["dx"]  # add the per plane offsets (dy,dx)

        f_0 = np.load(os.path.join(root, "suite2p", "plane%d" % n, "F.npy"))[iscell]
        f_neu0 = np.load(os.path.join(root, "suite2p", "plane%d" % n, "Fneu.npy"))[
            iscell
        ]
        f_0 = f_0 - 0.7 * f_neu0

        # compute spks0 with deconvolution
        if tlags[n] < 0:
            f_0[:, 1:] = (1 + tlags[n]) * f_0[:, 1:] + (-tlags[n]) * f_0[:, :-1]
        else:
            f_0[:, :-1] = (1 - tlags[n]) * f_0[:, :-1] + tlags[n] * f_0[:, 1:]

        f_0 = dcnv.preprocess(
            f_0.copy(),
            ops["baseline"],
            ops["win_baseline"],
            ops["sig_baseline"],
            ops["fs"],
            ops["prctile_baseline"],
        )
        spks0 = dcnv.oasis(f_0, ops["batch_size"], ops["tau"], ops["fs"])

        spks0 = spks0.astype("float32")
        iplane = np.concatenate((iplane, n * np.ones(len(stat0),)))
        stat = np.concatenate((stat, stat0), axis=0)
        if spks.shape[1] > spks0.shape[0]:
            spks0 = np.concatenate(
                (
                    spks0,
                    np.zeros(
                        (spks0.shape[0], spks.shape[1] - spks0.shape[1]), "float32"
                    ),
                ),
                axis=1,
            )
        spks = np.concatenate((spks, spks0), axis=0)
        ypos = np.concatenate((ypos, ypos0), axis=0)
        xpos = np.concatenate((xpos, xpos0), axis=0)

        print(f"plane {n}, neurons: {len(xpos0)}")

    print(f"total neurons {len(spks)}")

    xpos = xpos / 0.75
    ypos = ypos / 0.5

    return spks, stat, xpos, ypos, iplane


def get_neurons_atframes(timeline, spks):
    """
    Get the neurons at each frame, and the subset of stimulus before the recording ends.

    Parameters
    ----------
    spks : array
        Spikes of the neurons.
    Timeline : array
        Timeline of the experiment.

    Returns
    -------
    neurons_atframes : array
        Neurons at each frame.
    subset_stim: array
        Stimuli before recording ends
    """
    _, nt = spks.shape
    tlag = 1  # this is the normal lag between frames and stimuli
    istim = timeline["stiminfo"].item()["istim"]
    frame_start = timeline["stiminfo"].item()["frame_start"]
    frame_start = np.array(frame_start).astype("int")
    frame_start0 = frame_start + tlag
    ix = frame_start0 < nt
    frame_start0 = frame_start0[ix]
    neurons_atframes = spks[
        :, frame_start0
    ]  # sample the neurons at the stimulus frames
    subset_stim = istim[ix]
    return neurons_atframes, subset_stim


def get_tuned_neurons(neurons_atframes, subset_stim):
    """
    Gets the tuned neurons by computing the average of response for half of the two first categories presented,
    and then correlating to the average of the neurons for the other half.

    Parameters
    ----------
    neurons_atframes : array
        Neurons at each frame.
    subset_stim: array
        Stimuli before recording ends
    samples_per_category: int
        Number of samples per category.
    n_categories: int
        Number of categories to use.
    Returns
    -------
    csig : array
        (correlation between stimulus subsets) consistency of the neurons
    """
    unique_stimuli = np.unique(subset_stim)
    avg_response = np.zeros(
        (2, len(unique_stimuli), neurons_atframes.shape[0]), "float32"
    )
    k = 0
    s2 = np.zeros((2, len(unique_stimuli), neurons_atframes.shape[0]), "float32")
    for stim in unique_stimuli:
        ix = np.where(subset_stim == stim)[0]
        if len(ix) >= 2:
            avg_response[0, k] = neurons_atframes[:, ix[0]]
            avg_response[1, k] = neurons_atframes[:, ix[1]]
            k += 1
    avg_response = avg_response[:, :k]
    ss = avg_response - np.mean(avg_response, 1)[:, np.newaxis, :]
    ss = ss / np.mean(ss ** 2, 1)[:, np.newaxis, :] ** 0.5

    csig = (ss[0] * ss[1]).mean(0)
    return avg_response, csig


def get_decoder_stimuli(
    neurons_atframes, subset_stim, samples_per_category=4, train_exemplar=0,
):
    """
    Builds the decoder input array

    Parameters:
    ----------
    neurons_atframes : array
        Neurons at each frame.
    subset_stim: array
        Stimuli before recording ends
    samples_per_category: int
        Number of samples per category.
    train_exemplar: int
        Selected category to use as training
    Returns:
    ----------
    decoder_stimuli : array
        Decoder input array
    """
    unique_stimuli = np.unique(subset_stim)
    n_categories = len(unique_stimuli)
    cats_idx = np.arange(0, n_categories, samples_per_category)
    _, stimulus_counts = np.unique(subset_stim, return_counts=True)
    nreps = np.min(
        stimulus_counts[
            cats_idx[train_exemplar] : cats_idx[train_exemplar + 1]
            + samples_per_category
        ]
    )
    decoder_stimuli = np.zeros(
        (
            cats_idx[train_exemplar + 1] + samples_per_category,
            neurons_atframes.shape[0],
            nreps,
        )
    )
    for stimuli in unique_stimuli[
        : cats_idx[train_exemplar + 1] + samples_per_category
    ]:
        selected_stimuli = np.where(subset_stim == stimuli)[0][:nreps]
        decoder_stimuli[stimuli - 1] = neurons_atframes[:, selected_stimuli]
    X_train = decoder_stimuli[:, :, ::2]
    X_test = decoder_stimuli[:, :, 1::2]
    return X_train, X_test


class DprimeDecoder:

    """
    Implements a naive decoder based on the dprime separation of the neuron responses
    """

    def __init__(self, samples_per_category=4, train_exemplar=0, threshold=0.5, decisionrule="optimal"):
        DprimeDecoder.threshold = threshold
        DprimeDecoder.samples_per_category = samples_per_category
        DprimeDecoder.train_exemplar = train_exemplar
        DprimeDecoder.mu_ = None
        DprimeDecoder.sd_ = None
        DprimeDecoder.dprime_ = None
        DprimeDecoder.neurons_abvtresh_ = None
        DprimeDecoder.spop_ = None
        DprimeDecoder.clf_boundary_ = None
        DprimeDecoder.score_ = None
        DprimeDecoder.neurons_blwtresh_ = None 
        DprimeDecoder.decisionrule = decisionrule

    def middle_rule(self,spop_train):
        spop_train = spop_train.reshape(2,-1)
        trsh = (spop_train[0].min()+spop_train[1].max())/2
        #trsh = ((spop_train[0]+spop_train[1])/2).mean()
        return trsh

    def optimal_solution(self,spop_train):
        spop_train = spop_train.reshape(-1,1)
        y= np.concatenate((np.ones(int(spop_train.shape[0]/2)),np.zeros(int(spop_train.shape[0]/2))))
        train_min = spop_train.min()
        train_max = spop_train.max()
        threshold = np.arange(train_min,train_max,0.0001)
        trsh = []
        scre = []
        for t in threshold:
            pred = spop_train>t
            acc = accuracy_score(y,pred) * 100
            trsh.append(t)
            scre.append(acc)
        trsh = np.array(trsh)
        scre = np.array(scre)
        amax = np.argmax(scre)
        return trsh[amax]

    def fit(self, X, iplane, zstack=2):
        """
        Trains with even trials
        """
        self.mu_ = X.mean(-1)
        self.sd_ = X.std(-1)
        self.dprime_ = (
            2
            * (
                self.mu_[:self.samples_per_category]
                - self.mu_[self.samples_per_category:]
            )
            / (
                self.sd_[:self.samples_per_category]
                + self.sd_[self.samples_per_category:]
            )
        )
        self.neurons_abvtresh_ = self.dprime_[self.train_exemplar] > self.threshold
        self.neurons_blwtresh_ = -self.dprime_[self.train_exemplar] > self.threshold
        if zstack == 1:
            ix1 = (self.dprime_[self.train_exemplar] > self.threshold) * (iplane < 10)
            ix2 = (-self.dprime_[self.train_exemplar] > self.threshold) * (iplane < 10)
        else:
            ix1 = (self.dprime_[self.train_exemplar] > self.threshold) * (
                iplane >= 10
            )  # the last ten ROIs are in the top plane
            ix2 = (-self.dprime_[self.train_exemplar] > self.threshold) * (iplane >= 10)
        spop_train = X[:, ix1, :].mean(1) - X[:, ix2, :].mean(1)
        #spop_train = spop_train.reshape(2, -1)
        #y= np.concatenate((np.ones(spop_train.shape[1]),np.zeros(spop_train.shape[1])))
        spop_train = np.concatenate((spop_train[self.train_exemplar,:],spop_train[self.train_exemplar+self.samples_per_category,:]))
        spop_train = spop_train.reshape(-1,1)
        if self.decisionrule == "optimal":
            self.clf_boundary_ = self.optimal_solution(spop_train)
        else: 
            self.clf_boundary_ = self.middle_rule(spop_train) #get boundary with train trials
        return self



    def score(self, avg_test_reps = True):
        """
        Computes occuracy on test trials over average of repeats
        """
        if avg_test_reps:
            pred = np.mean(self.spop_, -1) > self.clf_boundary_
            y = np.concatenate((np.ones(self.samples_per_category), np.zeros(self.samples_per_category)))
            self.score_ = accuracy_score(y, pred) * 100
        else:
            pred = self.spop_ > self.clf_boundary_
            pred = pred.reshape(-1,1)
            y = np.concatenate((np.ones(int(pred.shape[0]/2)), np.zeros(int(pred.shape[0]/2))))
            self.score_ = accuracy_score(y, pred) * 100


        return self.score_

    def test(self, X, iplane, zstack=1):
        """
        Tests with odd trials
        """
        if zstack == 1:
            ix1 = (self.dprime_[self.train_exemplar] > self.threshold) * (iplane < 10)
            ix2 = (-self.dprime_[self.train_exemplar] > self.threshold) * (iplane < 10)
        else:
            ix1 = (self.dprime_[self.train_exemplar] > self.threshold) * (
                iplane >= 10
            )  # the last ten ROIs are in the top plane
            ix2 = (-self.dprime_[self.train_exemplar] > self.threshold) * (iplane >= 10)
        # spop is the decoder applied to test trials
        self.spop_ = X[:, ix1, :].mean(1) - X[:, ix2, :].mean(1)
        return self.spop_


def load_exp_info(db):
    """
    Loads the experiment info from the database

    Parameters:
    ----------
    db : dict
        Dictionary with the database entries
    Returns:
    ----------
    Timeline : dict
        Dictionary with the experiment info
    """
    mname, datexp, blk = db["mname"], db["datexp"], db["blk"]
    root = os.path.join("Z:/data/PROC", mname, datexp, blk)

    fname = "Timeline_%s_%s_%s" % (mname, datexp, blk)
    fnamepath = os.path.join(root, fname)

    Timeline = io.loadmat(fnamepath, squeeze_me=True)["Timeline"]

    return Timeline


def baselining(ops, tlag, F, Fneu):
    """
    Baseline the neural data before deconvolution

    Parameters:
    ----------
    ops : dict
        Dictionary with the experiment info
    tlag : int
        Time lag for the deconvolution
    F : array
        Deconvolved fluorescence
    Fneu : array
        Neurophil fluorescence
    Returns:
    ----------
    F : array
        Baselined deconvolved fluorescence
    """
    F = preprocess(F, Fneu, ops["win_baseline"], ops["sig_baseline"], ops["fs"])
    # F = dcnv.preprocess(F, ops['baseline'], ops['win_baseline'], ops['sig_baseline'],
    #                   ops['fs'], ops['prctile_baseline'])
    if tlag < 0:
        F[:, 1:] = (1 + tlag) * F[:, 1:] + (-tlag) * F[:, :-1]
    else:
        F[:, :-1] = (1 - tlag) * F[:, :-1] + tlag * F[:, 1:]
    return F


def preprocess(F, Fneu, win_baseline, sig_baseline, fs):
    """
    Preprocess the fluorescence data

    Parameters:
    ----------
    F : array
        Deconvolved fluorescence
    Fneu : array
        Neurophil fluorescence
    baseline : int
        Baseline for the fluorescence
    win_baseline : int
        Window for the baseline
    sig_baseline : int
        Sigma for the baseline
    fs : int
        Sampling rate
    Returns:
    ----------
    F : array
        Preprocessed deconvolved fluorescence
  """
    win = int(win_baseline * fs)

    Flow = ndimage.gaussian_filter(F, [0.0, sig_baseline])
    Flow = ndimage.minimum_filter1d(Flow, win)
    Flow = ndimage.maximum_filter1d(Flow, win)
    F = F - 0.7 * Fneu
    Flow2 = ndimage.gaussian_filter(F, [0.0, sig_baseline])
    Flow2 = ndimage.minimum_filter1d(Flow2, win)
    Flow2 = ndimage.maximum_filter1d(Flow2, win)

    Fdiv = np.maximum(10, Flow.mean(1))
    F = (F - Flow2) / Fdiv[:, np.newaxis]

    return F


### Load the neural data
def load_neurons(db, dual_plane=True, baseline=True):
    """
    Loads the neural data from the database

    Parameters:
    ----------

    db : dict
        Dictionary with the database entries
    dual_plane : Boolean
        Dual plane flag indicates whether the data is from the dual plane or not
    Baseline : Boolean
        Baseline flag indicates whether the data is preproceded or not
    Returns:
    ----------
    spks : array
        Spike matrix
        
    """
    mname, datexp, blk = db["mname"], db["datexp"], db["blk"]
    root = os.path.join("Z:/data/PROC", mname, datexp, blk)
    ops = np.load(
        os.path.join(root, "suite2p", "plane0", "ops.npy"), allow_pickle=True
    ).item()

    if dual_plane:
        tlags = np.linspace(0.2, -0.8, ops["nplanes"] // 2 + 1)[:-1]
        tlags = np.hstack((tlags, tlags))
        tlags = tlags.flatten()
    else:
        tlags = np.linspace(0.2, -0.8, ops["nplanes"] + 1)[:-1]
    print(tlags.shape)

    spks = np.zeros((0, ops["nframes"]), np.float32)
    stat = np.zeros((0,))
    iplane = np.zeros((0,))
    xpos, ypos = np.zeros((0,)), np.zeros((0,))

    for n in range(ops["nplanes"]):
        ops = np.load(
            os.path.join(root, "suite2p", "plane%d" % n, "ops.npy"), allow_pickle=True
        ).item()

        stat0 = np.load(
            os.path.join(root, "suite2p", "plane%d" % n, "stat.npy"), allow_pickle=True
        )
        ypos0 = np.array([stat0[n]["med"][0] for n in range(len(stat0))])
        xpos0 = np.array([stat0[n]["med"][1] for n in range(len(stat0))])

        ypos0 += ops["dy"]
        xpos0 += ops["dx"]

        if baseline:
            F = np.load(os.path.join(root, "suite2p", "plane%d" % n, "F.npy"))
            Fneu = np.load(os.path.join(root, "suite2p", "plane%d" % n, "Fneu.npy"))
            F = baselining(ops, tlags[n], F, Fneu)
            spks0 = dcnv.oasis(F, ops["batch_size"], ops["tau"], ops["fs"])
        else:
            spks0 = np.load(
                os.path.join(root, "suite2p", "plane%d" % n, "spks.npy"),
                allow_pickle=True,
            )

        spks0 = spks0.astype("float32")
        if spks.shape[1] > spks0.shape[0]:
            spks0 = np.concatenate(
                (spks0, np.zeros((spks0.shape[0], spks.shape[1] - spks0.shape[1]))),
                axis=1,
            )
        spks = np.concatenate((spks, spks0.astype("float32")), axis=0)
        ypos = np.concatenate((ypos, ypos0), axis=0)
        xpos = np.concatenate((xpos, xpos0), axis=0)
        iplane = np.concatenate((iplane, n * np.ones(len(stat0),)))
        stat = np.concatenate((stat, stat0), axis=0)

        print("plane %d, " % n, "neurons: %d" % len(xpos0))

    print("total neurons %d" % len(spks))

    return spks, xpos, ypos, iplane, stat

def get_stim_class_and_samples_ix(subset_stim, n_categories=8,samples_per_cat=4):
    """
    Gets the samples and repeats for the specified number of categories, and samples per category
    also returns the cats_idx containing the idx from a new category starts
    """
    total_samples = n_categories * samples_per_cat
    _, nc = np.unique(subset_stim, return_counts = True)
    nc = nc[:total_samples]
    nreps = np.min(nc)
    for exemplar in range(total_samples):
        if exemplar == 0:
            stim_idx = np.expand_dims(np.where(subset_stim==exemplar+1)[0][:nreps],axis=0)
        else:
            stim_idx= np.append(stim_idx, np.expand_dims(np.where(subset_stim==exemplar+1)[0][:nreps],axis=0),axis=0)
    cats_idx = np.arange(0, total_samples, samples_per_cat)
    print(f"{cats_idx.shape[0]} categories, {stim_idx.shape[0]} exemplars, {stim_idx.shape[1]} repeats")
    return cats_idx, stim_idx

def select_neurons_by_stimsubset(neurons_atframes, stim_idx, cats_idx, category_tostart, exemplars_to_take):
    """
    Returns the neurons at a given stim start category, and the following n stim exemplars
    """
    selected_neurons = np.zeros((exemplars_to_take, neurons_atframes.shape[0], stim_idx.shape[1]))
    for exemplar in range(exemplars_to_take):
        selected_neurons[exemplar] = neurons_atframes[:, stim_idx[cats_idx[category_tostart]+exemplar]]
    return selected_neurons