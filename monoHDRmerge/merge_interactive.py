import io
from typing import Optional
import math
from typing_extensions import Literal
from pydantic import BaseModel, Field, ConfigDict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import cv2


class coords2D(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    x: np.ndarray
    y: np.ndarray


class Range(BaseModel):
    min: int
    max: int


class mergeOptions(BaseModel):
    relative_exposure: Optional[dict] = None
    inclusion_boundary_width: float = Field(ge=0, le=1, default=0.2)
    relative_offset: float = Field(ge=-1, le=0.99, default=0)
    horizontal_offset: float = Field(ge=-1, le=1, default=0)
    bracketed_exposure_relation: Literal["linear", "sigmoid"] = "linear"
    activation_function: Literal["linear", "normal"] = "normal"
    exposure_filename_map: dict = None


def _clean_options(images: dict, options: mergeOptions):
    # TODO make sure the keys in relative exposure cover the images

    # Figure out relative exposure and neutral exposure if not defined
    if not options.relative_exposure:
        options.relative_exposure = {
            name: i - (len(images) - 1) / 2 for i, name in enumerate(images.keys())
        }

    # Check that each exposure is unique
    exposures = list(options.relative_exposure.values())
    if len(set(exposures)) < len(exposures):
        raise Exception("exposures should be unique")

    # This map is used often, so let's define it once
    options.exposure_filename_map = {
        value: key for key, value in options.relative_exposure.items()
    }

    return options


def _central_points_from_relational_function(
    x: np.ndarray, y: np.ndarray, options: mergeOptions
) -> dict:
    # Exposures sorted in inverse order; the higher the exposure the lower the
    # pixels values it should contribute
    sorted_exposures = np.sort(list(options.relative_exposure.values()))[::-1]
    linear_diff = np.array(list(range(len(sorted_exposures)))) / (
        len(sorted_exposures) - 1
    )

    # Get the central points
    central_points = {}
    for i, exposure in enumerate(sorted_exposures):
        x_index = (np.abs(y - linear_diff[i])).argmin()
        central_points[exposure] = np.round(x[x_index])

    return central_points


def linear_relation(x_range: Range, options: mergeOptions) -> coords2D:
    """Uses a linear relation between brackets to find the central points for the
    contribution function on the x-axis for each image.

    Args:
        x_range (Range): min and max of x
        options (mergeOptions)

    Returns:
        coords2D: x,y
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


def sigmoidal_relation(x_range: Range, options: mergeOptions) -> coords2D:
    """Uses a sigmoidal relation between brackets to find the central points for the
    contribution function on the x-axis for each image.

    Args:
        x_range (Range): min and max of x
        options (mergeOptions)

    Returns:
        coords2D: x,y
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


def linear_contribution(
    center_point: int, x_range: Range, options: mergeOptions
) -> coords2D:
    width = (x_range.max - x_range.min) * options.inclusion_boundary_width
    x = np.array(range(x_range.min, x_range.max + 1))
    y = -abs(((x - center_point) / (width / 2))) + 1

    # y now goes from -inf to 1, we only want between 0 and 1
    y[y < 0] = 0

    return coords2D(x=np.array(x), y=np.array(y))


def normal_function_contribution(
    center_point: int, x_range: Range, options: mergeOptions
) -> coords2D:
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


def get_weight_by_pixel_value(images: dict, options: mergeOptions) -> dict:
    dtype_info = np.iinfo(list(images.values())[0].dtype)
    x_range = Range(min=dtype_info.min, max=dtype_info.max)

    # First get the center points of the contribution function per image
    if options.bracketed_exposure_relation.lower() == "linear":
        center_points = linear_relation(x_range, options)
    elif options.bracketed_exposure_relation.lower() == "sigmoid":
        center_points = sigmoidal_relation(x_range, options)

    # Now define per image contribution by pixel value
    if options.activation_function.lower() == "normal":
        pixel_value_contributions = {
            exposure: normal_function_contribution(center_point, x_range, options)
            for exposure, center_point in center_points.items()
        }
    else:
        pixel_value_contributions = {
            exposure: linear_contribution(center_point, x_range, options)
            for exposure, center_point in center_points.items()
        }

    return pixel_value_contributions


def get_image_weight_maps(images: dict, options: mergeOptions) -> dict:
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


def merge(images: dict, options: mergeOptions):
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
    with io.BytesIO() as buff:
        figure.savefig(buff, format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    width, height = figure.canvas.get_width_height()
    return data.reshape((int(height), int(width), -1))


def plot_contribution(images: dict, options: mergeOptions) -> np.ndarray:
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
        weighted_image = cv2.resize(
            np.round(image * weight),
            dsize=(np.array(image.shape)[::-1] / 4).astype(int),
        )

        axis[i, 0].imshow(weighted_image, cmap="gray", vmin=0, vmax=65535)
        axis[i, 0].set_title(filename)

        # Histogram with y-range 0-1
        counts, bins = np.histogram(image.ravel(), bins=256)
        axis[i, 1].hist(bins[:-1], bins, weights=counts / counts.max())

        # Contribution function
        axis[i, 1].plot(contribution.x, contribution.y)

        axis[i, 1].set_title(f"relative exposure: {exposure}")

    # Return as np array
    return figure_to_array(f)


from load_and_save_images import load_images

plot_contribution(
    load_images(
        [
            "/home/bens/Pictures/scanned negatives/hdrtest/scan_01_-0.5.tif",
            "/home/bens/Pictures/scanned negatives/hdrtest/scan_01.tif",
            "/home/bens/Pictures/scanned negatives/hdrtest/scan_01_0.5.tif",
        ]
    ),
    options=mergeOptions(
        inclusion_boundary_width=0.2,
        relative_offset=0,
        horizontal_offset=0,
        bracketed_exposure_relation="sigmoid",
        activation_function="normal",
        # relative_exposure={
        #     "scan_01_-0.5.tif": 0,
        # },
    ),
)


# merged = merge(
#     load_images(
#         [
#             "/home/bens/Pictures/scanned negatives/hdrtest/scan_01_-0.5.tif",
#             "/home/bens/Pictures/scanned negatives/hdrtest/scan_01.tif",
#             "/home/bens/Pictures/scanned negatives/hdrtest/scan_01_0.5.tif",
#         ]
#     ),
#     options=mergeOptions(),
# )
