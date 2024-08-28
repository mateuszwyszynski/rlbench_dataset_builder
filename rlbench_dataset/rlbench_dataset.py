from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import pickle
from PIL import Image


class RLBenchDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('6.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
      '3.0.0': 'Observation with joint positions.',
      '4.0.0': 'Pick and lift task',
      '5.0.0': 'Pick and lift task with train and val splits',
      '6.0.0': 'Pick and lift task with train and val splits and only one variation',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rlbench_generated_dataset_path = "data/place_shape_in_shape_sorter"
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'proprio': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='6x robot joint angles + gripper state',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action for joints in one arm + gripper state of the next step',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path=self.rlbench_generated_dataset_path, train=True),
            'val': self._generate_examples(path=self.rlbench_generated_dataset_path, train=False),
        }

    def _generate_examples(self, path, train=True, train_size=0.9) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_episode(episode_path, variation_description):

            low_dim_obs_path = episode_path + "/low_dim_obs.pkl"

            with open(low_dim_obs_path, 'rb') as file:
                low_dim_obs = pickle.load(file)

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []

            for i, step in enumerate(low_dim_obs):
                # compute Kona language embedding
                language_embedding = self._embed([variation_description])[0].numpy()

                front_image_path = episode_path + f"/front_rgb/{i}.png"
                wrist_image_path = episode_path + f"/wrist_rgb/{i}.png"

                front_image = Image.open(front_image_path)
                wrist_image = Image.open(wrist_image_path)

                if i == len(low_dim_obs) - 1:
                    next_gripper_open = step["gripper_open"]
                else:
                    next_gripper_open = low_dim_obs[i+1]["gripper_open"]

                action = np.concatenate([step["joint_velocities"], [next_gripper_open]], axis=-1, dtype=np.float32)
                proprio = np.concatenate([step["joint_positions"], [step["gripper_open"]]], axis=-1, dtype=np.float32)

                episode.append({
                    'observation': {
                        'image': np.array(front_image),
                        'wrist_image': np.array(wrist_image),
                        'proprio': proprio,
                    },
                    'action': action,
                    'discount': 1.0,
                    'reward': float(i == (len(low_dim_obs) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(low_dim_obs) - 1),
                    'is_terminal': i == (len(low_dim_obs) - 1),
                    'language_instruction': variation_description,
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path + variation_description, sample

        variations_paths = glob.glob(path + "/variation*")

        for variation in variations_paths:
            variation_descriptions_path = variation + "/variation_descriptions.pkl"
            with open(variation_descriptions_path, 'rb') as file:
                variation_descriptions = pickle.load(file)

            variation_description_index = 0

            episodes_paths = glob.glob(variation + "/episodes/episode*")

            if train:
                episodes_paths = episodes_paths[:int(len(episodes_paths) * train_size)]
            else:
                episodes_paths = episodes_paths[int(len(episodes_paths) * train_size):]

            for episode_path in episodes_paths:
                yield _parse_episode(episode_path, variation_descriptions[variation_description_index])
                variation_description_index = (variation_description_index + 1) % len(variation_descriptions)



        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # paths = []
        # variations_paths = glob.glob(path + "/variation*")
        # for variation in variations_paths:
        #     variation_descriptions_path = variation + "/variation_descriptions.pkl"
        #     with open(variation_descriptions_path, 'rb') as file:
        #         variation_descriptions = pickle.load(file)
        #     for variation_description in variation_descriptions:
        #         episodes_paths = glob.glob(variation + "/episodes/episode*")
        #         for episode_path in episodes_paths:
        #             paths.append((episode_path, variation_description))
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(paths)
        #         | beam.Map(_parse_episode)
        # )