import gym
import custom_gym_env

def test_reset_and_step():
    env = gym.make('MountainCarContinuous-v1')
    env.reset()
    action = [0.5]
    next_state, reward, done, info = env.step(action)
    pass

