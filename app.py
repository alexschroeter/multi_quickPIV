from dataclasses import dataclass
from arkitekt import register
import time
from mikro.api.schema import (
    RepresentationFragment,
    from_xarray
)

from juliacall import Main as jl
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class PIVResult:
    u : np.ndarray
    v : np.ndarray

@register
def multi_quickPIV(image1 : RepresentationFragment,
                   image2 : RepresentationFragment,
                   step = (32,32,32),
                   interSize = (32,32,32),
                   searchMargin = (10,10,10),
                   threshold = (1000),
                   ) -> ImageFile:
    """
    This function performs a multithreaded Particle Image Velocimitry (PIV) on two images.
    The results are returned in form of a displacment vector for each interrogation window.
    So two images of dimension 100x100 with an intersize of 32x32 will result in a 3x3 grid of vectors.

    Parameters
    ----------
    image1 : RepresentationFragment
        Primary image which the PIV will be performed on

    image2 : RepresentationFragment
        Secondary image which the PIV will be performed on

    intersize : tuple
        Size of the interrogation window in pixels    
    
    step : tuple
        Step size between interrogation windows in pixels (windows can overlap)

    searchMargin : tuple
        Search margin around the interrogation window in pixels

    threshold : int
        Threshold for the correlation coefficient

    Returns
    -------
    tuple : (np.ndarray, np.ndarray)
        tuple whose first element is the PIV vector field, and the second element is the signal-to-noise ratio
    """
    

    # load the data of image1 and image2 and convert it to numpy array
    img1 = image1.data.sel(c=0, t=0).transpose(*"zxy").data.compute()
    img2 = image2.data.sel(c=0, t=0).transpose(*"zxy").data.compute()
    # for testing purposes we could also generate random data
    # img1 = np.random.rand(100, 100)
    # img2 = np.random.rand(100, 100)    

    # run PIV
    jl.seval("using multi_quickPIV")
    pivparams = jl.multi_quickPIV.setPIVParameters( interSize=jl.Array(interSize), searchMargin=jl.Array(searchMargin), step=jl.Array(step), threshold=jl.Array(threshold) )
    VF, SN = jl.multi_quickPIV.PIV( jl.Array(img1), jl.Array(img2), pivparams )

    # to vizualize the results we need to manipulte the data
    np_u = np.array(VF)[0]
    np_v = np.array(VF)[1]
    vfsize = np_u.shape

    # Creating the x and y coordinates for the quiver plot
    step   = np.array(jl.multi_quickPIV._step( pivparams )[1:2])
    isize  = np.array(jl.multi_quickPIV._isize( pivparams )[1:2])
    xgrid = np.array([ (x)*step[1] + isize[1]//2 for y in range(vfsize[0]) for x in range(vfsize[1]) ]).reshape(vfsize)
    ygrid = np.array([ (y)*step[0] + isize[0]//2 for y in range(vfsize[0]) for x in range(vfsize[1]) ]).reshape(vfsize)

    # Plotting
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("img1")
    plt.imshow(img1, cmap="Reds")

    plt.subplot(1, 3, 2)
    plt.title("img2")
    plt.imshow(img2, cmap="Blues")

    plt.subplot(1, 3, 3)
    plt.title("PIV")
    plt.imshow(img1, cmap="Reds")
    plt.imshow(img2, alpha=0.5, cmap="Blues")
    plt.quiver(xgrid, ygrid, np_v, np_u * -1, color="red", scale=50)
    plt.savefig("plot.png")

    file = open("plot.png")
    return file

@register
def plotPIV(PIVResult : PIVResult,
            ) -> RepresentationFragment:
    """
    This function plots the PIV result

    ToDo:
    - Add the image data to the PIVResult dataclass
    - pivparams need either be infered or stored
    - how to store the plot?

    Parameters

    PIVResult : PIVResult
        The result of the PIV calculation
    
    Returns
    -------
    RepresentationFragment
    """

    # Things that need to be infered by the PIVResult result provided
    # img1 and img2 inputs
    # pivparams (window size, step size, search margin, threshold)

    VF = PIVResult[0]
    SN = PIVResult[1]
    U = VF[ 0, :, : ]
    V = VF[ 1, :, : ]
    vfsize = U.shape

    # Creating the x and y coordinates for the quiver plot
    step   = np.array(jl.multi_quickPIV._step( pivparams )[1:2])
    isize  = np.array(jl.multi_quickPIV._isize( pivparams )[1:2])
    xgrid = np.array([ (x)*step[1] + isize[1]//2 for y in range(vfsize[0]) for x in range(vfsize[1]) ]).reshape(vfsize)
    ygrid = np.array([ (y)*step[0] + isize[0]//2 for y in range(vfsize[0]) for x in range(vfsize[1]) ]).reshape(vfsize)

    # Plotting
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("img1")
    plt.imshow(img1, cmap="Reds")

    plt.subplot(1, 3, 2)
    plt.title("img2")
    plt.imshow(img2, cmap="Blues")

    plt.subplot(1, 3, 3)
    plt.title("PIV")
    plt.imshow(img1, cmap="Reds")
    plt.imshow(img2, alpha=0.5, cmap="Blues")
    plt.quiver(xgrid, ygrid, np_v, np_u * -1, color="red", scale=50)
    return from_xarray(plot)
