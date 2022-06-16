"""
utility functions for the analysis of the data
"""
import os
from scipy import io, stats
import numpy as np
from suite2p.extraction import dcnv


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
    database.append({"mname": mname, "expdate": expdate, "block": blk})
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


def get_tuned_neurons(
    neurons_atframes, subset_stim, samples_per_category=4, n_categories=2
):
    """
    Gets the tuned neurons by computing the average of response for half of the stimuli presented,
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
    total_samples = n_categories * samples_per_category
    avg_response = np.zeros((2, total_samples, neurons_atframes.shape[0]), "float32")
    for stimuli in unique_stimuli[:total_samples]:
        selected_stimuli = np.where(subset_stim == stimuli)[0]
        even_idx = selected_stimuli[0::2]
        odd_idx = selected_stimuli[1::2]
        assert (
            len(selected_stimuli) >= 2
        ), "Not enough stimuli to compute the mean for this category"
        avg_response[0, stimuli - 1] = neurons_atframes[:, even_idx].mean(-1)
        avg_response[1, stimuli - 1] = neurons_atframes[:, odd_idx].mean(-1)
    z_avg_response = stats.zscore(avg_response, axis=1)
    csig = (z_avg_response[0] * z_avg_response[1]).mean(0)
    return csig


def get_decoder_stimuli(
    neurons_atframes, subset_stim, samples_per_category=4, selected_category=0,
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
    selected_category: int
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
            cats_idx[selected_category] : cats_idx[selected_category + 1]
            + samples_per_category
        ]
    )
    decoder_stimuli = np.zeros(
        (
            cats_idx[selected_category + 1] + samples_per_category,
            neurons_atframes.shape[0],
            nreps,
        )
    )
    for stimuli in unique_stimuli[
        : cats_idx[selected_category + 1] + samples_per_category
    ]:
        selected_stimuli = np.where(subset_stim == stimuli)[0][:nreps]
        decoder_stimuli[stimuli - 1] = neurons_atframes[:, selected_stimuli]
    return decoder_stimuli


class DprimeDecoder:

    """
    Implements a naive decoder based on the dprime separation of the neuron responses
    """
    def __init__(
        self, n_categories, samples_per_category=4, selected_category=0, threshold=0.5
    ):
        DprimeDecoder.threshold = threshold
        DprimeDecoder.n_categories = n_categories
        DprimeDecoder.samples_per_category = samples_per_category
        DprimeDecoder.selected_category = selected_category
        DprimeDecoder.mu_ = None
        DprimeDecoder.sd_ = None
        DprimeDecoder.dprime_ = None
        DprimeDecoder.neurons_abvtresh_ = None
        DprimeDecoder.spop_ = None

    def fit(self, X):
        """
        Trains with even trials
        """
        categories = np.arange(
            0, self.n_categories * self.samples_per_category, self.samples_per_category
        )
        cats_idx = (
            categories[self.selected_category],
            categories[self.selected_category + 1],
        )
        self.mu_ = X[:, :, ::2].mean(-1)
        self.sd_ = X[:, :, ::2].std(-1)
        self.dprime_ = (
            2
            * (self.mu_[cats_idx[0] : cats_idx[1]] - self.mu_[cats_idx[1] :])
            / (self.sd_[cats_idx[0] : cats_idx[1]] + self.sd_[cats_idx[1] :])
        )
        self.neurons_abvtresh_ = self.dprime_[self.selected_category] > self.threshold
        return self

    def test(self, X, iplane, zstack=1):
        """
        Tests with odd trials
        """
        if zstack == 1:
            ix1 = (self.dprime_[self.selected_category] > self.threshold) * (
                iplane < 10
            )
            ix2 = (-self.dprime_[self.selected_category] > self.threshold) * (
                iplane < 10
            )
        else:
            ix1 = (self.dprime_[self.selected_category] > self.threshold) * (
                iplane >= 10
            )  # the last ten ROIs are in the top plane
            ix2 = (-self.dprime_[self.selected_category] > self.threshold) * (
                iplane >= 10
            )
        # spop is the decoder applied to test trials
        self.spop_ = X[:, ix1, 1::2].mean(1) - X[:, ix2, 1::2].mean(1)
        return self
