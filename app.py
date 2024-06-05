from dataclasses import dataclass
from arkitekt import register
import time
from mikro.api.schema import (
    RepresentationFragment,
    from_xarray,
)

from juliacall import Main as jl
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class PIVResult:
    VF : np.ndarray
    SN : np.ndarray

@register
def multi_quickPIV(image1 : RepresentationFragment,
                   image2 : RepresentationFragment,
                #    step = (32,32,32),
                #    interSize = (32,32,32),
                #    searchMargin = (10,10,10),
                #    threshold = (1000),
                   ) -> PIVResult:
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
    print("starting")
    # load the data of image1 and image2 and convert it to numpy array
    img1 = image1.data.sel(c=0, t=0).data.compute()
    img2 = image2.data.sel(c=0, t=0).data.compute()
    # for testing purposes we could also generate random data
    # img1 = np.random.rand(100, 100)
    # img2 = np.random.rand(100, 100)    

    # run PIV
    print(jl.seval("Threads.nthreads()"))
    jl.seval("using multi_quickPIV")
    #pivparams = jl.multi_quickPIV.setPIVParameters( interSize=jl.Array(interSize), searchMargin=jl.Array(searchMargin), step=jl.Array(step), threshold=jl.Array(threshold) )
    # VF, SN = jl.multi_quickPIV.PIV( jl.Array(img1), jl.Array(img2), pivparams )
    VF, SN = jl.multi_quickPIV.PIV( jl.Array(img1), jl.Array(img2))

    print("Computed PIV successfully")
    print("VF: ", np.array(VF))
    print("SN: ", np.array(SN))
    return PIVResult(VF=np.array(VF), SN=np.array(SN))

@register
def plotPIV(image1 : RepresentationFragment,
            image2 : RepresentationFragment,
            PIVResult : PIVResult,
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
    step   = (32, 32)
    isize  = (32, 32)
    xgrid = np.array([ (x)*step[1] + isize[1]//2 for y in range(vfsize[0]) for x in range(vfsize[1]) ]).reshape(vfsize)
    ygrid = np.array([ (y)*step[0] + isize[0]//2 for y in range(vfsize[0]) for x in range(vfsize[1]) ]).reshape(vfsize)

    # load the data of image1 and image2 and convert it to numpy array
    img1 = image1.data.sel(c=0, t=0).data.compute()
    img2 = image2.data.sel(c=0, t=0).data.compute()

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
    plt.quiver(xgrid, ygrid, V, U * -1, color="red", scale=50)
    plt.savefig("piv.png")

    with open("piv.png", "rb") as f:
        plot = ImageFile(data=f.read())

    return from_xarray(plot)

@register
def example_PIV() -> None:
    """
    This function is an example of how to use the PIV functions
    """
    import urllib.request
    gif_ = urllib.request.urlretrieve("https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExY2FlOXJ6dHJjem5xNHZmaTVkMDdmMHZkanZvbGx1bDZ4b2d6M25sYSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/26xBuTV58WNJRlhLi/giphy.gif", "file.gif")
    from PIL import Image
    gif = Image.open('file.gif')

    frames = []
    # Loop through each frame in the GIF
    try:
        while True:
            # Convert the frame to an array and append to the list
            frame_array = np.array(gif.convert('L')) # Convert to grayscale
            frames.append(frame_array)

            # Move to the next frame
            gif.seek(gif.tell() + 1)
    except EOFError:
        # End of the sequence
        pass

    frames = np.array(frames)
    img1 = frames[6, :, :]
    img2 = frames[7, :, :]

    jl.seval("using multi_quickPIV")
    VF, SN = jl.multi_quickPIV.PIV( jl.Array(img1), jl.Array(img2))

    """
    We create a plot of the two Input Images and the resulting PIV vectors between them
    """
    VF = np.array(VF)
    SN = np.array(SN)

    U = VF[ 0, :, : ]
    V = VF[ 1, :, : ]
    vfsize = U.shape

    # Creating the x and y coordinates for the quiver plot
    step   = (32, 32)
    isize  = (32, 32)
    xgrid = np.array([ (x)*step[1] + isize[1]//2 for y in range(vfsize[0]) for x in range(vfsize[1]) ]).reshape(vfsize)
    ygrid = np.array([ (y)*step[0] + isize[0]//2 for y in range(vfsize[0]) for x in range(vfsize[1]) ]).reshape(vfsize)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Input Image 1 (Frame 6)")
    plt.imshow(img1, cmap="Reds")

    plt.subplot(1, 3, 2)
    plt.title("Input Image 2 (Frame 7)")
    plt.imshow(img2, cmap="Blues")

    plt.subplot(1, 3, 3)
    plt.title("PIV between Frame 6 and 7")
    plt.imshow(img1, cmap="Reds")
    plt.imshow(img2, alpha=0.5, cmap="Blues")
    plt.quiver(xgrid, ygrid, V, U * -1, color="red", scale=50)
    plt.savefig("example_arkitekt_piv.png")

    """
    We compare the results of the original PIV and the Arkitekt implementation
    """
    ground_truth = np.load("example_groundtruth.npz")
    diff = np.abs(VF, ground_truth)
    plt.clf()
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(diff[0], cmap='hot', interpolation='nearest')
    plt.title("U")
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.colorbar(label='Absolute Difference')

    plt.subplot(1, 2, 2)
    plt.imshow(diff[1], cmap='hot', interpolation='nearest')
    plt.title("V")
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.colorbar(label='Absolute Difference')

    plt.savefig("diff.png")
