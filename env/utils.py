import os

try:
    from grid2op.Episode import EpisodeReplay

    _CAN_USE = True
except ImportError:
    # cannot use the save_log_gif function
    _CAN_USE = False


def save_log_gif(path_log, res, gif_name=None):
    """
    Output a gif named (by default "episode.gif") that is the replay of the episode in a gif format,
    for each episode in the input.

    Parameters
    ----------
    path_log: ``str``
        Path where the log of the agents are saved.

    res: ``list``
        List resulting from the call to `runner.run`

    gif_name: ``str``
        Name of the gif that will be used.

    """
    if not _CAN_USE:
        raise RuntimeError("Cannot use the \"save_log_gif\" function as the "
                           "\"from grid2op.Episode import EpisodeReplay\" cannot be imported")

    init_gif_name = gif_name
    ep_replay = EpisodeReplay(path_log)
    for _, chron_name, cum_reward, nb_time_step, max_ts in res:
        if gif_name is None:
            gif_name = chron_name
        gif_path = os.path.join(path_log, chron_name, gif_name)
        print("Creating {}.gif".format(gif_name))
        ep_replay.replay_episode(episode_id=chron_name, gif_name=gif_name, display=False)
        print("Wrote {}.gif".format(gif_path))
        gif_name = init_gif_name