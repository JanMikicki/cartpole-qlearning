# cartpole-qlearning

Trying out traditional Q-learning (table) and Deep Q-Learning on the cartpole problem from gym envs (https://gym.openai.com/envs/CartPole-v1/) .

Cartpole can be controlled using 3 actions - left, right, nothing. The goal is to keep the pole upright, obviously. The episode ends when the pole tips over too much or it reaches some time limit.

Traditional method discretized (is this a word?), the state space into 20 x 20 x 20 x 20 table (4 since there are 4 observable states, 20 since why not). For each cell there are also 3 actions, so it's a (20, 20, 20, 20, 3) shape matrix in the end. It then updates this table according to qlearning formula.

It turned out pretty good. Around the 23k mark the epsilon reaches 0, meaning the exploration stage is over. After 25k it pretty much maxes out every time. (reward == duration of keeping the pole upright)

![I like cake.](ooh_wee.png?raw=true "Title")

DQN on the other hand turned out meh. Now, this in contrast uses a picture of a cartpole as an input, not just the 4 states. It stared out promising, but didn't go anywhere. Orange line is a rolling avereage here. 

![I like cake.](cartpole_2000.png?raw=true "Title")

One possible solution I found is that the neural net is "forgetting", explained in this post here: https://stackoverflow.com/a/54238556
There is a target network here though, so I'm not sure as to why it's so bad. It does however take more time to train than the traditional qlearning, so it is possible that it would start getting better after many more episodes (I just didn't have the patience.)

DQN is taken from this pytorch tutorial (https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) while the tradidtional one is based on a sentdex tutorial (linked in code).
