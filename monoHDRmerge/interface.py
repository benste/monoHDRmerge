from collections import defaultdict
import os

import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

import load_and_save_images
from merge_interactive import mergeOptions, plot_contribution, merge, figure_to_array

max_images_raw_images = 256


class gradioDataLink(object):
    """Need something to shuffle around data for objects in the
    UI where it cannot be determined beforehand how many there are.

    Since inputs and outputs to callbacks are static in gradio, with
    a theoritical limit of 256 images that would mean we'd need a function
    with 256 inputs. Thus this class
    """

    defaults = {"relative_exposure": 0}

    def __init__(self) -> None:
        self._data = defaultdict(dict)

        # set defaults
        for key, default in self.defaults.items():
            self._data[key] = defaultdict(lambda: default)

    @property
    def data(self) -> dict:
        return self._data

    def update_relative_exposure(self, label: str, exposure: float) -> None:
        self._data["relative_exposure"].update({label: exposure})


data_link = gradioDataLink()


def _create_relative_exposure_inputs():
    relative_exposure_labels, relative_exposure_values = ([], [])
    with gr.Blocks():
        instructions = gr.Markdown(
            "### Set Relative Exposure Difference Between Brackets ->",
            visible=False,
        )
        left_column = gr.Column(scale=4)
        right_column = gr.Column(scale=1)

        # file basename and the box to set their relative exposure
        for _ in range(max_images_raw_images):
            with left_column:
                relative_exposure_labels.append(
                    gr.Textbox(
                        value=None,
                        visible=False,
                        show_label=False,
                        interactive=False,
                    )
                )

            with right_column:
                relative_exposure_values.append(
                    gr.Number(
                        label="relative exposure",
                        visible=False,
                        show_label=False,
                        interactive=True,
                        value=data_link.defaults["relative_exposure"],
                    )
                )

                relative_exposure_values[-1].input(
                    data_link.update_relative_exposure,
                    inputs=[
                        relative_exposure_labels[-1],
                        relative_exposure_values[-1],
                    ],
                )

    return (
        instructions,
        relative_exposure_labels,
        relative_exposure_values,
    )


def _show_relative_exposure_options(image_locations: list):
    """Update the visibility and label of components that allow to set relative exposure

    Args:
        image_locations (list): _description_

    Returns:
        _type_: _description_
    """
    filenames = sorted([os.path.basename(path) for path in image_locations])
    label_updates, exposure_updates = ([], [])
    labels = filenames + [None] * (max_images_raw_images - len(filenames))
    for i in range(max_images_raw_images):
        label_updates.append(gr.update(visible=i < len(filenames), value=labels[i]))
        exposure_updates.append(gr.update(visible=i < len(filenames)))

    return (
        [gr.update(visible=True), gr.update(visible=True)]
        + label_updates
        + exposure_updates
    )


def merge_exposures(
    image_locations: list,
    bracketed_exposure_relation,
    activation_function,
    inclusion_boundary_width,
    horizontal_offset,
    relative_offset,
):
    images = load_and_save_images.load_images(image_locations)
    relative_exposure = {
        image: data_link.data["relative_exposure"][image] for image in images
    }
    options = mergeOptions(
        **{
            "relative_exposure": relative_exposure,
            "bracketed_exposure_relation": bracketed_exposure_relation,
            "activation_function": activation_function,
            "inclusion_boundary_width": inclusion_boundary_width,
            "horizontal_offset": horizontal_offset,
            "relative_offset": relative_offset,
        }
    )

    # Contribution information per bracket
    contribution_information = plot_contribution(images, options)

    # Merged image
    merged_image = merge(images, options)

    # Merged image histogram
    f, axis = plt.subplots(1)
    counts, bins = np.histogram(merged_image.ravel(), bins=256)
    axis.hist(bins[:-1], bins, weights=counts / counts.max())
    merged_image_histogram = figure_to_array(f)

    return [contribution_information, merged_image, merged_image_histogram]


with gr.Blocks() as mergeInterface:
    with gr.Row():
        with gr.Column(scale=4):
            # Upload images
            files_input = gr.File(file_count="multiple")

        # Components to set the relative exposure of each image.
        # Initially hidden until files are uploaded
        (
            relative_exposure_instructions,
            relative_exposure_labels,
            relative_exposure_values,
        ) = _create_relative_exposure_inputs()

    with gr.Row():
        # Once images have been uploaded and exposures defined, interactive merge can start
        start_interactive_merge_button = gr.Button(
            "start interactive merging", visible=False
        )

    with gr.Row():
        # Options changing how images are merged
        with gr.Column():
            with gr.Row():
                bracketed_exposure_relation = gr.Dropdown(
                    choices=["linear", "sigmoid"],
                    value="linear",
                    multiselect=False,
                    label="relationship between brackets",
                )
                activation_function = gr.Dropdown(
                    choices=["linear", "normal"],
                    value="normal",
                    multiselect=False,
                    label="activation function",
                )
            inclusion_boundary_width = gr.Slider(
                minimum=0, maximum=1, value=0.2, label="activation function width"
            )
            horizontal_offset = gr.Slider(
                minimum=-1, maximum=1, value=0, label="midpoint horizontal offset"
            )
            relative_offset = gr.Slider(
                minimum=-1, maximum=0.99, value=0, label="midpoint slope"
            )
            do_merge = gr.Button(value="MERGE", size="sm")
        # Here we'll show for each image what they contribute to the merged image
        with gr.Column():
            exposure_contribution_information = gr.Image(
                image_mode="L",
                show_download_button=False,
                interactive=False,
                label="exposure contribution per bracket",
            )

    # Here we'll show the resulting merged image
    with gr.Row():
        merged_image = gr.Image(
            image_mode="L",
            show_download_button=True,
            interactive=False,
            label="Merged HDR image",
        )

    # Here we'll show the resulting merged image histogram
    with gr.Row():
        merged_histogram = gr.Image(
            show_download_button=True, interactive=False, label="HDR image histogram"
        )

    # Only when bracketed images have been selected we'll show the options to set the
    # exposure per image
    files_input.change(
        _show_relative_exposure_options,
        inputs=[files_input],
        outputs=[relative_exposure_instructions, start_interactive_merge_button]
        + relative_exposure_labels
        + relative_exposure_values,
    )

    # This is all the options and outputs required for a merge
    # and to show the per bracket contribution information
    merge_inputs = [
        files_input,
        bracketed_exposure_relation,
        activation_function,
        inclusion_boundary_width,
        horizontal_offset,
        relative_offset,
    ]
    merge_outputs = [exposure_contribution_information, merged_image, merged_histogram]

    start_interactive_merge_button.click(
        merge_exposures,
        inputs=merge_inputs,
        outputs=merge_outputs,
    )
    do_merge.click(
        merge_exposures,
        inputs=merge_inputs,
        outputs=merge_outputs,
    )

mergeInterface.launch(share=False)
