import argparse
import tqdm
import importlib
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress debug warning messages
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import wandb
import plotly.express as px


WANDB_ENTITY = True
WANDB_PROJECT = "octo"


parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", help="name of the dataset to visualize")
args = parser.parse_args()

if WANDB_ENTITY is not None:
    render_wandb = True
    wandb.init(name="visualise_dataset", project="octo")
else:
    render_wandb = False


# create TF dataset
dataset_name = args.dataset_name
print(f"Visualizing data from dataset: {dataset_name}")
ds = tfds.load(dataset_name, split="train")
ds = ds.shuffle(100)

# visualize episodes
for i, episode in enumerate(ds.take(5)):
    images = []
    for step in episode["steps"]:
        images.append(step["observation"]["image"].numpy())
    image_strip = np.concatenate(images[::4], axis=1)
    caption = step["language_instruction"].numpy().decode() + " (temp. downsampled 4x)"

    if render_wandb:
        wandb.log({f"image_{i}": wandb.Image(image_strip, caption=caption)})
    else:
        plt.figure()
        plt.imshow(image_strip)
        plt.title(caption)

# visualize action and state statistics
actions, states, episode_lengths = [], [], []
for episode in tqdm.tqdm(ds.take(500)):
    episode_lengths.append([len(episode["steps"])])
    for step in episode["steps"]:
        actions.append(step["action"].numpy())
        states.append(step["observation"]["proprio"].numpy())

actions = np.array(actions)
states = np.array(states)
episode_lengths = np.array(episode_lengths)


def vis_stats(vector, tag):
    assert len(vector.shape) == 2

    vector_mean = vector.mean(0)
    vector_std = vector.std(0)
    vector_min = vector.min(0)
    vector_max = vector.max(0)

    n_elems = vector.shape[1]
    fig = plt.figure(tag, figsize=(5 * n_elems, 10))
    for elem in range(n_elems):
        plt.subplot(1, n_elems, elem + 1)
        plt.hist(vector[:, elem], bins=20)
        plt.title(
            f"mean={vector_mean[elem]}\nstd={vector_std[elem]}\nmin={vector_min[elem]}\nmax={vector_max[elem]}",
        )

    if render_wandb:
        wandb.log({tag: wandb.Image(fig)})


vis_stats(actions, "action_stats")
vis_stats(states, "state_stats")
vis_stats(episode_lengths, "episode_lengths")

if not render_wandb:
    plt.show()
