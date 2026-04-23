import matplotlib.pyplot as plt
import numpy as np


def plot_values(Q: np.array):
    if len(Q.shape) == 3:
        V = np.max(Q, axis=-1)
    elif len(Q.shape) == 2:
        V = Q
    else:
        raise ValueError(f"Invalid Q shape {Q.shape}")
    mask = np.flip(np.tri(V.shape[0], V.shape[0], k=-1), 1)
    V_masked = np.ma.array(V, mask=mask)
    f, ax = plt.subplots()
    im = ax.imshow(V_masked.T)
    bar = plt.colorbar(im)
    ax.set_xticks(np.arange(V.shape[0]))
    ax.set_yticks(np.arange(V.shape[1]))
    ax.invert_yaxis()
    ax.set_xlabel("Susceptible")
    ax.set_ylabel("Infected")
    ax.set_title("Value function")
    return f, ax


def plot_policy(Q: np.array):
    policy = np.argmax(Q, axis=-1)
    f, ax = plt.subplots()
    for m_s in range(policy.shape[0]):
        for m_i in range(policy.shape[1] - m_s):
            if policy[m_s, m_i] == 0:
                ax.plot(m_s, m_i, "x", color="red", label="confinement")
            else:
                ax.plot(m_s, m_i, "o", color="green", label="max exposure")
    ax.set_xticks(np.arange(Q.shape[0]))
    ax.set_yticks(np.arange(Q.shape[1]))
    ax.set_title("Social Optimum Policy")
    ax.set_xlabel("Susceptible")
    ax.set_ylabel("Infected")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    return f, ax
