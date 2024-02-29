# monoHDRmerge
Interactively merge exposure bracketed **monochromatic** images into an HDR image 

## How To Run

### Linux / MacOS
1. Make sure you have python ^3.10 active
2. Install dependencies using poetry by running ./scripts/.setup_poetry.sh
3. Run the UI using ./run.sh

A URI should be displayed in the terminal, probably http://127.0.0.1:7860/, open this in your browser


## How To Use

Load the bracketed images and assign the correct exposures to each image.
The exposures are relative to the other images, to [100, 200, 300] is the same as [0, 1, 2]

Now you can start the interactive merge. Left of the individual bracketed image preview are options
to change the distribution functions that determine which pixel value will contribute how much to the final 
image. 

In the individual bracketed image preview only what part of the image contributes to the final image is show,
together with the unmodified images' log histogram.


## Example
The final image will look flat, this is intended. 
This image can now be adjusted with your favorit image editor to create a more
pleasing image while retaining detail captured from each bracketed image.

bracket -2:
![bracket_-2](example_images/Image_-2.png)
bracket 1:
![bracket_-1](example_images/Image_-1.png)
bracket 0:
![bracket_0](example_images/Image_0.png)
bracket 1:
![bracket_1](example_images/Image_1.png)
bracket 2:
![bracket_2](example_images/Image_2.png)

Merged HDR image:
![mergedHDR](example_images/merged.png)
