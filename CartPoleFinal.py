import gym
import random
import math
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# hyper parameters
EPISODES = 500  # number of episodes
GAMMA = 0.95  # Q-learning discount factor
LEARNING_RATE = 0.01 #Stochastic grad descent
BATCH_SIZE = 32  # Q-learning batch size
MEMORY_SIZE = 10000
EPSILON = 0.9  #Initial Exploration Rate [0,1]. 1 --> fully random
EPSILON_MIN = 0.05 #Epsilon floor: 5% exploration
EPS_DECAY = 200  #Shifts episilon value


# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def add(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(torch.nn.Module):

    def __init__(self):
        torch.nn.Module.__init__(self)

        #2 fully connected linear layers. 24 to 32 to 2. 
        #how does number of layers affect our work?
        self.layer1 = torch.nn.Linear(in_features=4, out_features=24)
        self.layer2 = torch.nn.Linear(in_features=24, out_features=32)
        self.output = torch.nn.Linear(in_features=32, out_features=2)

    def forward(self, t):
        #forward pass tensor
        t = torch.nn.functional.relu(self.layer1(t))
        t = torch.nn.functional.relu(self.layer2(t))
        t = self.output(t)
        return t

class EpsilonGreedy():
    def __init__(self):
        self.start = EPSILON #.9
        self.end = EPSILON_MIN #.05
        self.decay = EPS_DECAY

    def get_exploration_rate(self, curr_step):
        #return 0.1
        return self.end + (self.start - self.end) * math.exp(-1 * curr_step / self.decay)
        #add annealing later

class Agent():
    def __init__(self, env):

        self.env = env
        self.model = DQN()
        if use_cuda:
            self.model.cuda()
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.strategy = EpsilonGreedy()
        self.optimizer = optim.Adam(self.model.parameters(), LEARNING_RATE)
        self.steps_done = 0
        self.episode_durations = []


    def select_action(self, state):
        #global steps_done
        sample = random.random()
        rate = self.strategy.get_exploration_rate(self.steps_done)
        self.steps_done += 1
        if sample > rate:
            with torch.no_grad():
                q_vals = self.model(Variable(state)).cpu().numpy()
                action_to_take = np.argmax(q_vals)
                return action_to_take
        else:
            return np.random.randint(2)

    def run_episode(self, e, environment):
        state = environment.reset()
        steps = 0
        while True:
            environment.render()
            action = self.select_action(FloatTensor([state]))
            next_state, reward, done, _ = environment.step(action) 
            self.memory.add((state, action, next_state, reward, done))

            self.learn() #place outside of loop for larger situations

            state = next_state
            steps += 1

            if done: 
                print("Episode: {0}/{1}, Score: {2}".format(e, EPISODES, steps))
                self.episode_durations.append(steps)
                self.plot_durations(self.episode_durations, "Score")
                break

    def learn(self):

        if len(self.memory) < BATCH_SIZE:
            return

        # random transition batch is taken from experience replay memory
        transitions = self.memory.sample(BATCH_SIZE)

        batch_state, batch_action, batch_next_state, batch_reward, dones = zip(*transitions)

        #32x4, 32x2, 32x4, 32x1, 32x1
        batch_state = np.asarray(batch_state)
        batch_action = np.asarray(batch_action)
        batch_next_state = np.asarray(batch_next_state)
        batch_reward = np.asarray(batch_reward)
        dones = np.asarray(dones, dtype=np.float64)

        #Tensor concatentations of 1,2,3,4th values in transition tuple
        batch_state = torch.from_numpy(batch_state).cuda().float()
        batch_action = torch.from_numpy(batch_action).cuda().long()
        batch_reward = torch.from_numpy(batch_reward).cuda().float()
        batch_next_state = torch.from_numpy(batch_next_state).cuda().float()
        batch_done = torch.from_numpy(dones).cuda().float()

        # current Q values is estimated by NN for all actions
        model_output = self.model(batch_state)
        
        one_hot_selection = F.one_hot(batch_action, 2)
        current_q_values = torch.sum(model_output * one_hot_selection, 1)

        # expected Q values are estimated from actions which gives maximum Q value
        max_next_q_values = self.model(batch_next_state).detach().max(1)[0] #

        discounted_values =  max_next_q_values * GAMMA
        mask_values = discounted_values * (1 - batch_done)

        expected_q_values = batch_reward + mask_values
        
        # loss is measured from error between current and newly expected Q values
        loss = F.mse_loss(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        
        # backpropagation of loss to NN
        loss.backward()
        self.optimizer.step()
        

    def plot_durations(self, l, title):
        plt.figure(2)
        plt.clf()
        durations_t = torch.FloatTensor(l)
        plt.title(title)
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        plt.pause(0.001)  # pause to update plot

        # take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        

if __name__ == '__main__':
    environment = gym.make('CartPole-v0')

    cartPoleSolver = Agent(environment)

    for e in range(EPISODES):
        cartPoleSolver.run_episode(e, environment)

    print('Complete')
    environment.render()
    environment.close()
    plt.ioff()
    plt.show()
