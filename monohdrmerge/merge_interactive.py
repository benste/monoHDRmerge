import io
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import gennorm
from typing_extensions import Literal


class coords2D(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    x: np.ndarray
    y: np.ndarray


class Range(BaseModel):
    min: int
    max: int


class mergeOptions(BaseModel):
    relative_exposure: dict = {}
    inclusion_boundary_width: float = Field(ge=0, le=1, default=0.2)
    relative_offset: float = Field(ge=-1, le=0.99, default=0)
    horizontal_offset: float = Field(ge=-1, le=1, default=0)
    bracketed_exposure_relation: Literal["linear", "sigmoid"] = "linear"
    activation_function: Literal["linear", "normal", "subbotin"] = "subbotin"
    exposure_filename_map: dict = {}


def _clean_options(images: dict, options: mergeOptions) -> mergeOptions:
    # Figure out relative exposures if not defined
    if not options.relative_exposure:
        options.relative_exposure = {
            name: i - (len(images) - 1) / 2 for i, name in enumerate(sorted(images.keys()))
        }

    # Check that each exposure is unique
    exposures = list(options.relative_exposure.values())
    if len(set(exposures)) < len(exposures):
        raise Exception("exposures should be unique")

    # check that the relative exposure cover the images
    if any(exposure not in images.keys() for exposure in options.relative_exposure.keys()):
        raise Exception("exposures do not cover all the images")

    # This map is used often, so let's define it once
    options.exposure_filename_map = {value: key for key, value in options.relative_exposure.items()}

    return options


def _central_points_from_relational_function(
    x: np.ndarray, y: np.ndarray, options: mergeOptions
) -> dict:
    # Exposures sorted in inverse order; the higher the exposure the lower the
    # pixels values it should contribute
    sorted_exposures = np.sort(list(options.relative_exposure.values()))[::-1]
    linear_diff = np.array(list(range(len(sorted_exposures)))) / (len(sorted_exposures) - 1)

    # Get the central points
    central_points = {}
    for i, exposure in enumerate(sorted_exposures):
        x_index = (np.abs(y - linear_diff[i])).argmin()
        central_points[exposure] = np.round(x[x_index])

    return central_points


def linear_relation(x_range: Range, options: mergeOptions) -> dict:
    """Uses a linear relation between brackets to find the central points for the
    contribution function on the x-axis for each image.

    Args:
        x_range (Range): min and max of x
        options (mergeOptions)

    Returns:
        dict: {exposure: center point on the x-axis}
    """
    if len(options.relative_exposure) == 1:
        # Simply take the center of the single image
        return {list(options.relative_exposure.values())[0]: np.round(x_range.max / 2)}

    # X start end end of the linear relation by taking relative offset into account
    x_start = round((options.relative_offset * (x_range.max - x_range.min)) / 2)
    x_end = x_range.max - x_start
    x = np.array(range(x_start, x_end))
    if len(x) < 1:
        x = np.array([x_start])

    # Add the horizontal offset
    x = x + options.horizontal_offset * x_range.max

    # Calculate y
    y = (x - x.min()) * (1 / (x.max() - x.min()))

    return _central_points_from_relational_function(x, y, options)


def sigmoidal_relation(x_range: Range, options: mergeOptions) -> dict:
    """Uses a sigmoidal relation between brackets to find the central points for the
    contribution function on the x-axis for each image.

    Args:
        x_range (Range): min and max of x
        options (mergeOptions)

    Returns:
        dict: {exposure: center point on the x-axis}
    """
    if len(options.relative_exposure) == 1:
        # Simply take the center of the single image
        return {list(options.relative_exposure.values())[0]: np.round(x_range.max / 2)}

    # X start end end of the linear relation by taking relative offset into account
    x_start = round((options.relative_offset * (x_range.max - x_range.min)) / 2)
    x_end = x_range.max - x_start
    x = np.array(range(x_start, x_end))
    if len(x) < 1:
        x = np.array([x_start])

    # Add the horizontal offset
    x = x + options.horizontal_offset * x_range.max

    # Calculate y
    x_width = x.max() - x.min()
    x_for_sigmoid = x - x.min()
    x_for_sigmoid = (x_for_sigmoid - x_width / 2) / (x_width / 10)  # range from -5 to 5
    y = 1 / (1 + np.exp(-x_for_sigmoid))

    # make sure y goes from 0 to 1
    y = y - y.min()
    y = y / y.max()

    return _central_points_from_relational_function(x, y, options)


def linear_contribution(center_point: int, x_range: Range, options: mergeOptions) -> coords2D:
    """activation function that goes up and down around the center point linearly

    Args:
        center_point (int): center of the activation
        x_range (Range): min and max x can be
        options (mergeOptions)

    Returns:
        coords2D: x and y of the activation
    """
    width = (x_range.max - x_range.min) * options.inclusion_boundary_width
    x = np.array(range(x_range.min, x_range.max + 1))
    y = -abs(((x - center_point) / (width / 2))) + 1

    # y now goes from -inf to 1, we only want between 0 and 1
    y[y < 0] = 0

    return coords2D(x=np.array(x), y=np.array(y))


def normal_function_contribution(
    center_point: int, x_range: Range, options: mergeOptions
) -> coords2D:
    """Normal distribution activation function around the center point

    Args:
        center_point (int): center of the activation
        x_range (Range): min and max x can be
        options (mergeOptions)

    Returns:
        coords2D: x and y of the activation
    """
    width = (x_range.max - x_range.min) * options.inclusion_boundary_width
    x = np.array(range(x_range.min, x_range.max + 1))
    standard_deviation = width / 3
    median = center_point
    y = (1 / (standard_deviation * math.sqrt(2 * math.pi))) * np.exp(
        -0.5 * ((x - median) / standard_deviation) ** 2
    )

    # Make sure y goes from 0 to 1
    y = y - y.min()
    y = y / y.max()

    return coords2D(x=np.array(x), y=np.array(y))


def subbotin_contribution(center_point: int, x_range: Range, options: mergeOptions) -> coords2D:
    """Subbotin distribution activation function around the center point

    Args:
        center_point (int): center of the activation
        x_range (Range): min and max x can be
        options (mergeOptions)

    Returns:
        coords2D: x and y of the activation
    """

    width = (x_range.max - x_range.min) * options.inclusion_boundary_width
    x = np.array(range(x_range.min, x_range.max + 1))

    # With a beta of 10 and the width ranging from -1.2 to 1.2 we get a good shape
    rv = gennorm(10)
    y = rv.pdf((x - center_point) / (width / 2 / 1.2))

    # Make sure y goes from 0 to 1
    y = y - y.min()
    y = y / y.max()

    # Make sure y goes from 0 to 1
    y = y - y.min()
    y = y / y.max()

    return coords2D(x=np.array(x), y=np.array(y))


def get_weight_by_pixel_value(images: dict, options: mergeOptions) -> dict:
    """For each image, the weight a pixel-value would contribute to the final image

    value = how white, since this only deals with monochromatic images

    Args:
        images (dict): {name: np.ndarray}
        options (mergeOptions)

    Returns:
        dict: {exposure: pixel value weight}
    """
    dtype_info = np.iinfo(list(images.values())[0].dtype)
    x_range = Range(min=dtype_info.min, max=dtype_info.max)

    # First get the center points of the contribution function per image
    if options.bracketed_exposure_relation.lower() == "linear":
        center_points = linear_relation(x_range, options)
    elif options.bracketed_exposure_relation.lower() == "sigmoid":
        center_points = sigmoidal_relation(x_range, options)

    # Now define per image contribution by pixel value
    activation_function_map = {
        "normal": normal_function_contribution,
        "linear": linear_contribution,
        "subbotin": subbotin_contribution,
    }
    activation_function = activation_function_map[options.activation_function.lower()]
    pixel_value_contributions = {
        exposure: activation_function(center_point, x_range, options)
        for exposure, center_point in center_points.items()
    }

    return pixel_value_contributions


def get_image_weight_maps(images: dict, options: mergeOptions) -> dict:
    """For each image, for each pixel, it's contributing weight

    Args:
        images (dict): {name: np.ndarray}
        options (mergeOptions)

    Returns:
        dict: {image name: weight map}
    """
    # get the pixel-value -> weight maps
    pixel_value_weight = get_weight_by_pixel_value(images, options)

    # Now map this to the image pixel coordinates
    weight_maps = {}
    for exposure, weight in pixel_value_weight.items():
        filename = options.exposure_filename_map[exposure]
        weight = weight.y[images[filename]]

        # Add to weights maps by filename
        weight_maps[filename] = weight

    return weight_maps


def merge(images: dict, options: mergeOptions) -> np.ndarray:
    """Merge bracketed images into an HDR image

    Args:
        images (dict): {name: np.ndarray}
        options (mergeOptions)

    Returns:
        np.ndarray: Merged image
    """
    options = _clean_options(images, options)

    # Get per image weight maps
    weight_maps = get_image_weight_maps(images, options)

    # Create a 3D matrix for the weights and images
    all_images = np.stack(list(images.values()))
    all_weights = np.stack([weight_maps[filename] for filename in list(images.keys())])
    all_weights = np.nan_to_num(all_weights)

    # get the weighted average per pixel
    merged_images = (all_images * all_weights).sum(axis=0) / all_weights.sum(axis=0)

    # return with correct type
    image_type = list(images.values())[0].dtype
    return np.nan_to_num(np.round(merged_images)).astype(image_type)


def figure_to_array(figure: Figure) -> np.ndarray:
    """transform a matplotlib figure to numpy array"""
    with io.BytesIO() as buff:
        figure.savefig(buff, format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    width, height = figure.canvas.get_width_height()
    return data.reshape((int(height), int(width), -1))


def plot_contribution(images: dict, options: mergeOptions) -> np.ndarray:
    """Plot for each bracketed exposure
    - The image with each pixel lightened or darkened depending on its contribution
    - A exposure histrogram with the activation function overlayed

    Args:
        images (dict): {name: np.ndarray}
        options (mergeOptions)

    Returns:
        np.ndarray: matplotlib subplot as numpy array
    """
    options = _clean_options(images, options)

    contribution_by_pixel_value = get_weight_by_pixel_value(images, options)
    weight_maps = get_image_weight_maps(images, options)

    # Sort the exposures to plot the least brightest at the top
    sorted_exposures = np.sort(list(options.relative_exposure.values()))

    f, axis = plt.subplots(max(len(images), 2), 2)
    for i, exposure in enumerate(sorted_exposures):
        filename = options.exposure_filename_map[exposure]
        contribution = contribution_by_pixel_value[exposure]
        image = images[filename]
        weight = weight_maps[filename]

        # apply weights to image and resize for faster rendering
        weighted_image = cv2.resize(  # type: ignore
            np.round(image * weight),
            dsize=(np.array(image.shape)[::-1] / 4).astype(int),
        )

        axis[i, 0].imshow(weighted_image, cmap="gray", vmin=0, vmax=65535)
        axis[i, 0].set_title(filename)

        # Log histogram with y-range 0-1
        counts, bins = np.histogram(image.ravel(), bins=256)
        counts = np.log(counts)
        counts = counts / counts.max()
        axis[i, 1].hist(bins[:-1], bins, weights=counts)

        # Contribution function
        axis[i, 1].plot(contribution.x, contribution.y)

        axis[i, 1].set_title(f"relative exposure: {exposure}")

    # Return as np array
    return figure_to_array(f)
