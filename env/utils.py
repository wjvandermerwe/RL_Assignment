import math

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import seaborn as sns

def visualize_action_scatter(episodes_data):
    """
    Visualize the actions as a scatter plot to show their frequency and variety during episodes.
    :param episodes_data: List of loaded EpisodeData objects.
    """
    plt.figure(figsize=(12, 8))
    for i, episode_data in enumerate(episodes_data):
        action_counts = {}
        actions = episode_data.actions

        # Count actions taken during the episode
        for act in actions:
            action_type = act.act_name  # Assuming `act.act_name` gives an identifier for the action type
            action_counts[action_type] = action_counts.get(action_type, 0) + 1

        # Create scatter plot data for the episode
        action_types = list(action_counts.keys())
        counts = list(action_counts.values())
        plt.scatter(action_types, counts, label=f"Episode {i}", alpha=0.6)

    plt.xlabel("Action Type")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.title("Scatter Plot of Actions Taken in Episodes")
    plt.legend()
    plt.grid(True)
    plt.savefig("actions_scatter.png")

def visualize_peak_load_heatmap(episodes_data, load_id=1):
    """
    Create a heatmap to show peak load occurrences over multiple episodes.
    :param episodes_data: List of loaded EpisodeData objects.
    :param load_id: ID of the load to visualize.
    """
    peak_loads = np.zeros((len(episodes_data), len(episodes_data[0].observations)))
    for i, episode_data in enumerate(episodes_data):
        for j, obs in enumerate(episode_data.observations):
            dict_ = obs.state_of(load_id=load_id)
            peak_loads[i, j] = dict_['p']

    plt.figure(figsize=(12, 8))
    sns.heatmap(peak_loads, cmap='hot', cbar=True)
    plt.xlabel("Timestep")
    plt.ylabel("Episode Index")
    plt.title(f"Heatmap of Load {load_id} Consumption Over Episodes")
    plt.savefig("peak_load_heatmap.png")