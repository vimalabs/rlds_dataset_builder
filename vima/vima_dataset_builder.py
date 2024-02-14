from typing import Iterator, Tuple, Any
import os
import pickle
from copy import deepcopy

import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from vima.conversion_utils import MultiThreadedDatasetBuilder

from einops import rearrange


TASK_MAPPING = {
    "follow_order": "follow_order",
    "manipulate_old_neighbor": "manipulate_old_neighbor",
    "novel_adj": "novel_adj",
    "novel_noun": "novel_noun",
    "pick_in_order_then_restore": "pick_in_order_then_restore",
    "rearrange": "rearrange",
    "rearrange_then_restore": "rearrange_then_restore",
    "rotate": "rotate",
    "same_profile": "same_shape",
    "scene_understanding": "scene_understanding",
    "simple_manipulation": "visual_manipulation",
    "sweep_without_exceeding": "sweep_without_exceeding",
    "twist": "twist",
}


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Generator of examples for each split."""

    views = ["top", "front"]

    def _parse_example(episode_path):
        """
        episode_path: /path_to_data_dir/task_name/episode_id
        """

        rgbs_all_views = {}
        for view in views:
            rgb_view_folder = os.path.join(episode_path, f"rgb_{view}")
            n_frames = len(
                [x for x in os.listdir(rgb_view_folder) if x.endswith(".jpg")]
            )
            rgbs = []
            for i in range(n_frames):
                img = np.array(
                    Image.open(os.path.join(rgb_view_folder, f"{i}.jpg")),
                    copy=True,
                    dtype=np.uint8,
                )
                rgbs.append(img)  # list of (H, W, C)
            rgbs_all_views[view] = rgbs
        segm_and_ee = pickle.load(open(os.path.join(episode_path, "obs.pkl"), "rb"))
        actions = pickle.load(open(os.path.join(episode_path, "action.pkl"), "rb"))
        traj_meta = pickle.load(
            open(os.path.join(episode_path, "trajectory.pkl"), "rb")
        )

        # construct some fields that are consistent across all steps
        # ====== segmentation_obj_info ======
        obj_id_to_info = traj_meta.pop("obj_id_to_info")
        segm_id, obj_name, texture_name = [], [], []
        for k, v in obj_id_to_info.items():
            segm_id.append(np.array(k, dtype=np.int64))
            obj_name.append(v["obj_name"])
            texture_name.append(v["texture_name"])
        segmentation_obj_info = {
            "segm_id": segm_id,
            "obj_name": obj_name,
            "texture_name": texture_name,
        }

        # ====== multimodal_instruction_assets ======
        prompt_assets = traj_meta.pop("prompt_assets")
        key_name, asset_type = [], []
        images, frontal_images, segmentations, frontal_segmentations = (
            [],
            [],
            [],
            [],
        )
        prompt_segmentation_obj_info = []
        for k, v in prompt_assets.items():
            key_name.append(k)
            asset_type.append(v["placeholder_type"])
            images.append(rearrange(v["rgb"]["top"], "c h w -> h w c"))
            frontal_images.append(rearrange(v["rgb"]["front"], "c h w -> h w c"))
            segmentations.append(v["segm"]["top"])
            frontal_segmentations.append(v["segm"]["front"])
            obj_info = v["segm"]["obj_info"]
            if v["placeholder_type"] == "object":
                obj_info = [obj_info]
            prompt_segmentation_obj_info.append(
                {
                    "segm_id": [
                        np.array(each_obj_info["obj_id"], dtype=np.int64)
                        for each_obj_info in obj_info
                    ],
                    "obj_name": [
                        each_obj_info["obj_name"] for each_obj_info in obj_info
                    ],
                    "texture_name": [
                        each_obj_info["obj_color"] for each_obj_info in obj_info
                    ],
                }
            )
        multimodal_instruction_assets = {
            "key_name": key_name,
            "asset_type": asset_type,
            "image": images,
            "frontal_image": frontal_images,
            "segmentation": segmentations,
            "frontal_segmentation": frontal_segmentations,
            "segmentation_obj_info": prompt_segmentation_obj_info,
        }

        num_steps = traj_meta["steps"]
        step_wise_data = []
        for t in range(num_steps):
            step_wise_data.append(
                {
                    "observation": {
                        "image": rgbs_all_views["top"][t],
                        "frontal_image": rgbs_all_views["front"][t],
                        "segmentation": segm_and_ee["segm"]["top"][t],
                        "frontal_segmentation": segm_and_ee["segm"]["front"][t],
                        "segmentation_obj_info": deepcopy(segmentation_obj_info),
                        "ee": segm_and_ee["ee"][t],
                    },
                    "action": {
                        "pose0_position": actions["pose0_position"][t],
                        "pose0_rotation": actions["pose0_rotation"][t],
                        "pose1_position": actions["pose1_position"][t],
                        "pose1_rotation": actions["pose1_rotation"][t],
                    },
                    "discount": 1.0,
                    "reward": float(t == (num_steps - 1)),
                    "is_first": t == 0,
                    "is_last": t == (num_steps - 1),
                    "is_terminal": t == (num_steps - 1),
                    "multimodal_instruction": traj_meta["prompt"],
                    "multimodal_instruction_assets": deepcopy(
                        multimodal_instruction_assets
                    ),
                }
            )
        # get relative file path in the format of task_name/episode_id
        task_name = os.path.basename(os.path.dirname(episode_path))
        episode_id = os.path.basename(episode_path)
        file_path = os.path.join(task_name, episode_id)

        episode_metadata = {
            "task": TASK_MAPPING[task_name],
            "file_path": file_path,
            "action_bounds": {
                "high": traj_meta["action_bounds"]["high"],
                "low": traj_meta["action_bounds"]["low"],
            },
            "end-effector type": traj_meta["end_effector_type"],
            "failure": traj_meta["failure"],
            "success": traj_meta["success"],
            "n_objects": traj_meta["n_objects"],
            "robot_components_seg_ids": [
                np.array(x, dtype=np.int64) for x in traj_meta["robot_components"]
            ],
            "seed": traj_meta["seed"],
            "num_steps": num_steps,
        }

        # create output data sample
        sample = {"steps": step_wise_data, "episode_metadata": episode_metadata}

        # if you want to skip an example for whatever reason, simply return None
        return file_path, sample

    # loop over all paths
    for sample in paths:
        yield _parse_example(sample)


class VIMADataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for VIMA dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    N_WORKERS = 10  # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = (
        100  # number of paths converted & stored in memory before writing to disk
    )
    # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
    # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = (
        _generate_examples  # handle to parse function from file paths to RLDS episodes
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # read data folder from environment variable
        self._raw_data_dir = os.environ.get("RAW_DATA_DIR", None)
        assert (
            self._raw_data_dir is not None
        ), "Please set the RAW_DATA_DIR environment variable."
        assert os.path.exists(
            self._raw_data_dir
        ), f"Data directory {self._raw_data_dir} does not exist."
        for task_name in TASK_MAPPING:
            assert os.path.exists(
                os.path.join(self._raw_data_dir, task_name)
            ), f"Task directory {task_name} does not exist."

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Tensor(
                                        shape=(128, 256, 3),
                                        dtype=np.uint8,
                                        doc="Topdown camera RGB observation.",
                                    ),
                                    "frontal_image": tfds.features.Tensor(
                                        shape=(128, 256, 3),
                                        dtype=np.uint8,
                                        doc="Frontal camera RGB observation.",
                                    ),
                                    "segmentation": tfds.features.Tensor(
                                        shape=(128, 256),
                                        dtype=np.uint8,
                                        doc="Topdown camera segmentation observation. "
                                        "The mapping between object segmentation ID and its information "
                                        "can be found in `obj_id_to_info`.",
                                    ),
                                    "frontal_segmentation": tfds.features.Tensor(
                                        shape=(128, 256),
                                        dtype=np.uint8,
                                        doc="Frontal camera segmentation observation. "
                                        "The mapping between object segmentation ID and its information "
                                        "can be found in `obj_id_to_info`.",
                                    ),
                                    "segmentation_obj_info": tfds.features.FeaturesDict(
                                        {
                                            "segm_id": tfds.features.Sequence(
                                                tfds.features.Scalar(
                                                    dtype=np.int64,
                                                    doc="The segmentation object ID.",
                                                ),
                                                doc="A list of segmentation object IDs.",
                                            ),
                                            "obj_name": tfds.features.Sequence(
                                                tfds.features.Text(
                                                    doc="The segmentation object name."
                                                ),
                                                doc="A list of segmentation object names.",
                                            ),
                                            "texture_name": tfds.features.Sequence(
                                                tfds.features.Text(
                                                    doc="The segmentation object's texture name."
                                                ),
                                                doc="A list of segmentation object's texture names.",
                                            ),
                                        },
                                        doc="Information about objects in the segmentation.",
                                    ),
                                    "ee": tfds.features.Tensor(
                                        shape=(),
                                        dtype=np.int64,
                                        doc="Indicate the end-effector's type. 0 for a suction cup, 1 for a spatula.",
                                    ),
                                }
                            ),
                            "action": tfds.features.FeaturesDict(
                                {
                                    "pose0_position": tfds.features.Tensor(
                                        shape=(3,),
                                        dtype=np.float32,
                                        doc="XYZ position for pick",
                                    ),
                                    "pose0_rotation": tfds.features.Tensor(
                                        shape=(4,),
                                        dtype=np.float32,
                                        doc="Quaternion rotation for pick",
                                    ),
                                    "pose1_position": tfds.features.Tensor(
                                        shape=(3,),
                                        dtype=np.float32,
                                        doc="XYZ position for place",
                                    ),
                                    "pose1_rotation": tfds.features.Tensor(
                                        shape=(4,),
                                        dtype=np.float32,
                                        doc="Quaternion rotation for place",
                                    ),
                                },
                                doc="Robot action, consists of two poses for pick and place.",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount if provided, default to 1.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward if provided, 1 on final step for demos.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on first step of the episode."
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on last step of the episode."
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "multimodal_instruction": tfds.features.Text(
                                doc="Multimodal Instruction, consists of both texts and images."
                            ),
                            "multimodal_instruction_assets": tfds.features.FeaturesDict(
                                {
                                    "key_name": tfds.features.Sequence(
                                        tfds.features.Text(
                                            doc="The key name that appears in the instruction and assets. "
                                            "For example, `frame_0` in the instruction "
                                            "'Stack objects in this order {frame_0} {frame_1} {frame_2}.'"
                                        ),
                                        doc="A list of key names that appears in the instruction and assets. "
                                        "The list length varies depending on the instruction.",
                                    ),
                                    "asset_type": tfds.features.Sequence(
                                        tfds.features.Text(
                                            doc="The type of the image. "
                                            "For example, a `scene` image or an `object` image."
                                        ),
                                        doc="A list of asset types that corresponds to the `key_name` list.",
                                    ),
                                    "image": tfds.features.Sequence(
                                        tfds.features.Tensor(
                                            shape=(128, 256, 3),
                                            dtype=np.uint8,
                                            doc="The top-down RGB image that appears in the multimodal instruction.",
                                        ),
                                        doc="A list of top-down RGB images that corresponds to the `key_name` list.",
                                    ),
                                    "frontal_image": tfds.features.Sequence(
                                        tfds.features.Tensor(
                                            shape=(128, 256, 3),
                                            dtype=np.uint8,
                                            doc="The frontal RGB image that appears in the multimodal instruction.",
                                        ),
                                        doc="A list of frontal RGB images that corresponds to the `key_name` list.",
                                    ),
                                    "segmentation": tfds.features.Sequence(
                                        tfds.features.Tensor(
                                            shape=(128, 256),
                                            dtype=np.uint8,
                                            doc="The top-down segmentation that appears in the multimodal instruction.",
                                        ),
                                        doc="A list of top-down segmentation images that "
                                        "corresponds to the `key_name` list.",
                                    ),
                                    "frontal_segmentation": tfds.features.Sequence(
                                        tfds.features.Tensor(
                                            shape=(128, 256),
                                            dtype=np.uint8,
                                            doc="The frontal segmentation that appears in the multimodal instruction.",
                                        ),
                                        doc="A list of frontal segmentation images that "
                                        "corresponds to the `key_name` list.",
                                    ),
                                    "segmentation_obj_info": tfds.features.Sequence(
                                        tfds.features.FeaturesDict(
                                            {
                                                "segm_id": tfds.features.Sequence(
                                                    tfds.features.Scalar(
                                                        dtype=np.int64,
                                                        doc="The segmentation object ID.",
                                                    ),
                                                    doc="A list of segmentation object IDs.",
                                                ),
                                                "obj_name": tfds.features.Sequence(
                                                    tfds.features.Text(
                                                        doc="The segmentation object name."
                                                    ),
                                                    doc="A list of segmentation object names.",
                                                ),
                                                "texture_name": tfds.features.Sequence(
                                                    tfds.features.Text(
                                                        doc="The segmentation object texture."
                                                    ),
                                                    doc="A list of segmentation object textures.",
                                                ),
                                            },
                                            doc="Information about objects in the segmentation.",
                                        ),
                                        doc="A list of object segmentation information that "
                                        "corresponds to the `key_name` list.",
                                    ),
                                },
                                doc="Assets for one multimodal instruction.",
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "task": tfds.features.Text(
                                doc="One of the tasks in VIMABench."
                            ),
                            "file_path": tfds.features.Text(
                                doc="Path to the original data file."
                            ),
                            "action_bounds": tfds.features.FeaturesDict(
                                {
                                    "high": tfds.features.Tensor(
                                        shape=(3,),
                                        dtype=np.float32,
                                        doc="Upper bound for xyz position.",
                                    ),
                                    "low": tfds.features.Tensor(
                                        shape=(3,),
                                        dtype=np.float32,
                                        doc="Lower bound for xyz position.",
                                    ),
                                },
                                doc="Action bounds for the task.",
                            ),
                            "end-effector type": tfds.features.Text(
                                doc="The type of the end-effector. Can be `suction` or `spatula`."
                            ),
                            "failure": tfds.features.Scalar(
                                dtype=np.bool_, doc="True if the task is failed."
                            ),
                            "success": tfds.features.Scalar(
                                dtype=np.bool_, doc="True if the task is successful."
                            ),
                            "n_objects": tfds.features.Scalar(
                                dtype=np.int64, doc="Number of objects in the task."
                            ),
                            "robot_components_seg_ids": tfds.features.Sequence(
                                tfds.features.Scalar(
                                    dtype=np.int64,
                                    doc="The segmentation ID corresponding to a robot part.",
                                ),
                                doc="A list of segmentation object IDs corresponding to robot parts.",
                            ),
                            "seed": tfds.features.Scalar(
                                dtype=np.int64,
                                doc="The seed used to generate the task and the demonstration.",
                            ),
                            "num_steps": tfds.features.Scalar(
                                dtype=np.int64,
                                doc="Number of action steps in the episode.",
                            ),
                        }
                    ),
                }
            )
        )

    def _split_paths(self):
        """Define filepaths for data splits."""
        paths = []
        # loop over all tasks
        for task_name in TASK_MAPPING:
            # list all folders
            all_folders = os.listdir(os.path.join(self._raw_data_dir, task_name))
            # trajectories are in folders with names that are integers
            all_folders = [f for f in all_folders if f.isdigit()]
            # loop over all trajectories
            for f_path in all_folders:
                f_path = os.path.join(self._raw_data_dir, task_name, f_path)
                paths.append(f_path)
        return {
            "train": paths,
        }
