from gym.envs.registration import register

register(
    id="MountainCarContinuous-v1",
    entry_point="normalized_continuous_mountain_car:Normalized_Continuous_MountainCarEnv",
    max_episode_steps=999,
    reward_threshold=90.0,
)
