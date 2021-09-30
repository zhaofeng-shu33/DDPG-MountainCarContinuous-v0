## Mountain Car Continuous problem 
How to run using old package
```
module load tensorflow/1.15.0-cpu
module load anaconda3/py3
```
python==3.6.8
keras==2.2.4
tensorflow==1.15.0
gym==0.19.0

### DDPG solving Openai Gym

Without any seed it can solve within 2 episodes but on average it takes 4-6
The Learner class have a plot_Q method that will plot some very usefull graphs to tweak the model

In the training, the learner act in the environment using `0.2*action + noise` where `action` is the local actor model, `noise` is an Ornstein-Uhlenbeck process and the number `0.2` is a chosen scalar to avoid the learner getting
stuck trying to use some bad actions (like +1 for all states, which can happen at first). After every epoch
the learner will be tested against the environment without any noise in the actions and if the average of the 
`n_solved` number of tests are above 90 the loop will break and the learner object will output the rewards and
steps of every episode.


### Project Instructions

1. Run the MountainCar.py file directly or import it on the notebook.

```
# This way you don't see much
> python MountainCar.py 
```

Or

```
> from MountainCar import MountainCar
> Learner = MountainCar()
> train_hist, test_hist, solved = Learner.run_model(max_epochs=20, n_solved = 1, verbose=1)

```

2. To test the final model

```
> _, test_hist, solved = Learner.run_model(max_epochs=20, n_solved = 1, verbose=-1)
```

3. Before running make sure to have the necessary modules


---------------------
#### References
This project was create while I was attending the Udacity's Machine Learning Engineer Nanodegree all files except `MountainCar.py` contains mostly tweaked boilerplate code of the DDPG model provided. You can probably find the originals [here](https://github.com/udacity/deep-reinforcement-learning)
