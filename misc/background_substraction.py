import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology


def remove_background(rgb, dataset):
    """
    Try to compute the mask between background and foreground
    Args:
        rgb: T,  H, W, C
    Returns:
        mask : T, H, W
    """

    rgb = ((rgb + 1) / 2)

    ## STEP 1 : Found most probable background
    backgrounds = [plt.imread(f"misc/plane_{i}{'_balls' if dataset == 'balls' else ''}.png")[..., :3] for i in
                   range(4)]

    error = [((rgb - b.reshape((1, 112, 112, 3))) ** 2).mean() for b in backgrounds]
    background = backgrounds[np.argmin(error)]

    ## STEP 2 : Remove and threshold
    difference = ((rgb - background.reshape((1, 112, 112, 3))) ** 2).sum(-1)
    difference = difference > 0.1

    for t in range(difference.shape[0]):
        difference[t] = skimage.morphology.binary_closing(difference[t])

    # fig, ax = plt.subplots(1, 3)
    # ax[0].imshow(rgb[0])
    # ax[1].imshow(background)
    # ax[2].matshow(difference[0])
    #
    # for a in ax:
    #    a.axis("off")
    # fig.tight_layout()
    #
    # for t in range(rgb.shape[0]):
    #    ax[0].clear()
    #    ax[2].clear()
    #    ax[1].clear()
    #    ax[0].axis("off")
    #    ax[2].axis("off")
    #    ax[1].axis("off")
    #    ax[0].imshow(rgb[t])
    #    ax[2].matshow(difference[t])
    #    ax[1].imshow(rgb[t] * difference[t].reshape((112, 112, 1)))
    #    plt.pause(0.001)
    return difference.reshape((difference.shape[0], 112, 112, 1))
