#importing the dependencies of the python files 
import numpy as np
import gym
import random

#make the environment for the Taxi game
env = gym.make("Taxi-v2")
env.render()

#creating the Q - table to know the rows and columns but we can calculate the action_size and state_size based on inbuilt commands
action_size = env.action_space.n
print("Action size", action_size)

state_size = env.observation_space.n
print("State size", state_size)

#creating the Qtable to store the Q values
qtable = np.zeros((state_size, action_size))
print(qtable)

#specify the hyperparameters which are used to train the algorithm
total_episodes = 50000     #total episodes
total_test_episodes = 100  #total test episodes
max_steps = 99  #Max Steps per episode

learning_rate = 0.7 #learning rate
gamma = 0.618 #discount rate

#Exploration parameters
epsilon = 1.0 #exploration rate
max_epsilon = 1.0 #Exploration probability at start
min_epsilon = 0.01 #Minimum exploration probability
decay_rate = 0.01 #Expotential decay rate for Exploration


# For life or until the learning is stopped 
for episode in range(total_episodes):
    #reset the environment
    state = env.reset()
    step = 0
    done = False

    for step in range(max_steps):
        #choose an action a in the current world state (s)
        ## first we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)

        # if this number > greater than epsilon --> Exploitation(taking the biggest Q-value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])

        #Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()

        #take the action (a) and observe the outcome state(s') and reward (r)
        new_state,reward, done, info = env.step(action)

        #Update QFunction(S, a) := Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

        #the new state is now the state
        state = new_state

        #if we have completed then finish the episode
        if done == True:
            break

    episode+= 1
    print("Episode Complete")

    #reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode) 



env.reset()
rewards = []

for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    print("*********************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
        env.render()
        #take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state, :])

        new_state, reward, done, info = env.step(action)

        total_rewards += reward

        if done:
            rewards.append(total_rewards)
            print ("Score", total_rewards)
            break
        
        state = new_state

env.close()
print("Score over time: " + str(sum(rewards)/total_test_episodes))

