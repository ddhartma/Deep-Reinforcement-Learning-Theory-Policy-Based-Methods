[image1]: assets/intro_policy_based_methods.png "image1"
[image2]: assets/policy_function_approx.png "image2"
[image3]: assets/more_policy_based_methods.png "image3"
[image4]: assets/hill_climbing.png "image4"
[image5]: assets/hill_climbing_algo.png "image5"
[image6]: assets/hill_climbing_opti_1.png "image6"
[image7]: assets/hill_climbing_opti_2.png "image7"
[image8]: assets/hill_climbing_opti_3.png "image8"
[image9]: assets/hill_climbing_opti_4.png "image9"
[image10]: assets/optimizations_black_box.png "image10"
[image11]: assets/hill_climbing_plot.png "image11"
[image12]: assets/hill_climbing_agent.png "image12"
[image13]: assets/cem_plot.png "image13"
[image14]: assets/cem_agent.png "image14"


# Deep Reinforcement Learning Theory - Policy Based Methods

## Content 
- [Introduction](#intro)
- [Policy Based Methods](#policy_based_methods)
    - [Policy Function Approximation](#Policy_Function_Approximation)
    - [More on the Policy](#More_on_the_Policy)
    - [Hill Climbing - Gradient Ascent](#hill_climbing)
    - [Hill Climbing Pseudocode](#hill_climbing_pseudo)
    - [Beyond Hill Climbing](#beyond_hill_climbing)
    - [More Black-Box Optimization](#black_box)
    - [Why Policy Based Methods?](#why_policy_based_methods)
- [Hill Climbing - Code](#hill_climbing_code)
- [Cross Entropy Method](#cross_entropy_method)
- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

## Introduction <a name="what_is_reinforcement"></a>
- Reinforcement learning is **learning** what to do — **how to map situations to actions** — so as **to maximize a numerical reward** signal. The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them. (Sutton and Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html))
- Deep reinforcement learning refers to approaches where the knowledge is represented with a deep neural network

### Overview:
- ***Policy-Based Methods***: methods such as 
    - hill climbing
    - simulated annealing
    - adaptive noise scaling
    - cross-entropy methods
    - evolution strategies

- ***Policy Gradient Methods***:
    - REINFORCE
    - lower the variance of policy gradient algorithms

- ***Proximal Policy Optimization***:
    - Proximal Policy Optimization (PPO), a cutting-edge policy gradient method

- ***Actor-Critic Methods***
    - how to combine value-based and policy-based methods
    - bringing together the best of both worlds, to solve challenging reinforcement learning problems

## Policy Based Methods <a name="policy_based_methods"></a> 
- With **value-based methods**, the agent uses its experience with the environment to maintain an estimate of the optimal action-value function. The optimal policy is then obtained from the optimal action-value function estimate.
- **Policy-based methods** directly learn the optimal policy, without having to maintain a separate value function estimate.

    ![image1]

## Policy Function Approximation <a name="Policy_Function_Approximation"></a> 
- In deep reinforcement learning, it is common to represent the policy with a neural network.
    - This network takes the environment state as **input**.
    - If the environment has discrete actions, the **output** layer has a node for each possible action and contains the probability that the agent should select each possible action.
- The weights in this neural network are initially set to random values. Then, the agent updates the weights as it interacts with (and learns more about) the environment.

    ![image2]

## More on the Policy <a name="More_on_the_Policy"></a> 
- Policy-based methods can learn either stochastic or deterministic policies, and they can be used to solve environments with either finite or continuous action spaces. 

    ![image3]

## Hill Climbing - Gradient Ascent<a name="hill_climbing"></a> 
- **Gradient ascent** is similar to gradient descent.
    - Gradient descent steps in the ***direction opposite the gradient***, since it wants to minimize a function.
    - Gradient ascent is otherwise identical, except we step in the **direction of the gradient**, to reach the maximum.

- **Hill climbing** is an iterative algorithm that can be used to find the weights &theta;  for an optimal policy.
    - At each iteration,
        - We slightly perturb the values of the current best estimate for the weights &theta;<sub>best</sub>, to yield a new set of weights.
        - These new weights are then used to collect an episode. If the new weights &theta;<sub>new</sub> resulted in higher return than the old weights, then we set &theta;<sub>best</sub> ← &theta;<sub>new</sub>.
        - **Objective function J=J(θ)**: Find **&theta;** that miximizes **J**. Find **argmax<sub>&theta;</sub>J(&theta;)** 

    ![image4]

- [Hill Climbing](https://en.wikipedia.org/wiki/Hill_climbing) is not just for reinforcement learning! It is a general optimization method that is used to find the maximum of a function.
        
## Hill Climbing Pseudocode <a name="hill_climbing_pseudo"></a> 
- Find the pseudocode for Hill Climbing below

    ![image5]



### What's the difference between G and J?
- for the same **&theta;** value  **G** will likely be different from episode to episode
- sampled return **G** 
- expected return **J**
- **G** is not a perfect estimate but often **good enough** for **J**


## Beyond Hill Climbing <a name="beyond_hill_climbing"></a> 
- Objective function can marked by a contaour plot

    ![image6]

- **Steepest ascent hill climbing** is a variation of hill climbing that chooses a small number of neighboring policies at each iteration and chooses the best among them.

    ![image7]
- **Simulated annealing** uses a pre-defined schedule to control how the policy space is explored, and gradually reduces the search radius as we get closer to the optimal solution.

    ![image8]

- **Adaptive noise scaling** decreases the search radius with each iteration when a new best policy is found, and otherwise increases the search radius.

    ![image9]

## More Black-Box Optimization <a name="black_box"></a>
- All of the algorithms in this lesson can be classified as **black-box optimization** techniques. Black-box refers to the fact that the way we evaluate **θ** is considered a black box.
- Further black-box optimizations:
    - The **cross-entropy method** iteratively suggests a small number of neighboring policies, and uses a small percentage of the best performing policies to calculate a new estimate.
    - The **evolution strategies** technique considers the return corresponding to each candidate policy. 
    - The policy estimate at the next iteration is a weighted sum of all of the candidate policies, where policies that got higher return are given higher weight.

    ![image10]
 

## Why Policy Based Methods? <a name="why_policy_based_methods"></a>
- There are three reasons why we consider policy-based methods:
    1. **Simplicity**: Policy-based methods directly get to the problem at hand (estimating the optimal policy), without having to store a bunch of additional data (i.e., the action values) that may not be useful.
    2. **Stochastic policies**: Unlike value-based methods, policy-based methods can learn true stochastic policies.
    3. **Continuous action spaces**: Policy-based methods are well-suited for continuous action spaces.

## Hill Climbing - Code <a name="hill_climbing_code"></a> 
- Open Jupyter notebook ```hill_climbing.ipynb```
    ### Import the Necessary Packages
    ```
    import gym
    import numpy as np
    from collections import deque
    import matplotlib.pyplot as plt
    %matplotlib inline

    !python -m pip install pyvirtualdisplay
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(1400, 900))
    display.start()

    is_ipython = 'inline' in plt.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()
    ```
    ### Define the Policy
    ```
    env = gym.make('CartPole-v0')
    print('observation space:', env.observation_space)
    print('action space:', env.action_space)

    class Policy():
        """ Define a Policy
        """
        def __init__(self, s_size=4, a_size=2):
            """ Initialize weights in the policy arbitrarily
                
                INPUTS:
                ------------
                    s_size - (int) size of state space
                    a_size - (int) size of action space
                
                OUTPUTS:
                ------------
                    no direct 
                    self.w (numpy array) rows: state_space, columns: action_space (4x2 matrix)
            
            """
            self.w = 1e-4*np.random.rand(s_size, a_size)  # weights for simple linear policy: state_space x action_space
            
        def forward(self, state):
            """ Create forward pass of neural network. Dot product of state vector with weight matrix
            
                INPUTS: 
                ------------
                    state (numpy array)
                
                OUTPUTS:
                ------------
                    output (numpy array) softmax classification output
            """
            x = np.dot(state, self.w)
            output = np.exp(x)/sum(np.exp(x))
            return output
        
        def act(self, state):
            """ Execute forward pass get best action for given state
        
                INPUTS:
                ------------
                    state (numpy array) of state values (4 values)
                
                OUTPUTS:
                ------------
                    action (numpy array) one dimensional, max likely action
            """
            probs = self.forward(state)
            #action = np.random.choice(2, p=probs) # option 1: stochastic policy
            action = np.argmax(probs)              # option 2: deterministic policy
            return action

    RESULTS:
    ------------
    observation space: Box(4,)
    action space: Discrete(2)
    ```
    ### Train the Agent with Stochastic Policy Search
    ```
    env = gym.make('CartPole-v0')
    env.seed(0)
    np.random.seed(0)

    policy = Policy()

    def hill_climbing(n_episodes=1000, max_t=1000, gamma=1.0, print_every=100, noise_scale=1e-2):
        """ Implementation of hill climbing with adaptive noise scaling.
            
            INPUTS:
            ------------
                n_episodes - (int) maximum number of training episodes
                max_t - (int) maximum number of timesteps per episode
                gamma - (float) discount rate
                print_every - (int) how often to print average score (over last 100 episodes)
                noise_scale - (float) standard deviation of additive noise
                
            OUTPUTS:
            ------------
                scores - (list) of accumulated rewards
        """
        scores_deque = deque(maxlen=100)
        scores = []
        best_R = -np.Inf
        best_w = policy.w
        for i_episode in range(1, n_episodes+1):
            rewards = []
            state = env.reset()
            for t in range(max_t):
                action = policy.act(state)
                state, reward, done, _ = env.step(action)
                rewards.append(reward)
                if done:
                    break 
            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))

            discounts = [gamma**i for i in range(len(rewards)+1)]
            R = sum([a*b for a,b in zip(discounts, rewards)])

            if R >= best_R: # found better weights
                best_R = R
                best_w = policy.w
                noise_scale = max(1e-3, noise_scale / 2)
                policy.w += noise_scale * np.random.rand(*policy.w.shape) 
            else: # did not find better weights
                noise_scale = min(2, noise_scale * 2)
                policy.w = best_w + noise_scale * np.random.rand(*policy.w.shape)

            if i_episode % print_every == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            if np.mean(scores_deque)>=195.0:
                print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
                policy.w = best_w
                break
            
        return scores
                
    scores = hill_climbing()

    RESULTS:
    ------------
    state:  [-0.04456399  0.04653909  0.01326909 -0.02099827]
    action: 0

    Episode 100	Average Score: 175.24
    Environment solved in 13 episodes!	Average Score: 196.21
    ```
    ### Plot the Scores
    ```
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    ```
    ![image11]

    ### Watch a smart agent
    ```
    env = gym.make('CartPole-v0')
    state = env.reset()
    img = plt.imshow(env.render(mode='rgb_array'))
    for t in range(200):
        action = policy.act(state)
        img.set_data(env.render(mode='rgb_array')) 
        plt.axis('off')
        display.display(plt.gcf())
        display.clear_output(wait=True)
        state, reward, done, _ = env.step(action)
        if done:
            break 

    env.close()
    ```
    ![image12]


## Cross Entropy Method <a name="cross_entropy_method"></a> 
- Open Jupyter notebook ```cross_entropy_method.ipynb```
    ### Import the Necessary Packages
    ```
    import gym
    import math
    import numpy as np
    from collections import deque
    import matplotlib.pyplot as plt
    %matplotlib inline

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable

    !python -m pip install pyvirtualdisplay
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(1400, 900))
    display.start()

    is_ipython = 'inline' in plt.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()
    ```
    ### Instantiate the Environment and Agent
    ```
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = gym.make('MountainCarContinuous-v0')
    env.seed(101)
    np.random.seed(101)

    print('observation space:', env.observation_space)
    print('action space:', env.action_space)
    print('  - low:', env.action_space.low)
    print('  - high:', env.action_space.high)

    class Agent(nn.Module):
        """ Define an MountainCarContinuous agent
        """
        def __init__(self, env, h_size=16):
            """ Initialize environment and neural network layer parameter
            
                INPUTS:
                ------------
                    env - (gym object)
                    h_size - (int) unit number of hidden layers 
                
                OUTPUTS:
                ------------
                    no direct
            """
            super(Agent, self).__init__()
            self.env = env
            # state, hidden layer, action sizes
            self.s_size = env.observation_space.shape[0]
            self.h_size = h_size
            self.a_size = env.action_space.shape[0]
            # define layers
            self.fc1 = nn.Linear(self.s_size, self.h_size)
            self.fc2 = nn.Linear(self.h_size, self.a_size)
            
        def set_weights(self, weights):
            """ Set weights for neural network, Input -- Hidden -- Output --> 2 weight sets
            
                INPUTS:
                ------------
                    weights - (numpy array) one dimensional, 65 values
                
                OUTPUTS:
                ------------
                    no direct
            """
            s_size = self.s_size
            h_size = self.h_size
            a_size = self.a_size
            
            # separate the weights for each layer
            fc1_end = (s_size*h_size)+h_size
            fc1_W = torch.from_numpy(weights[:s_size*h_size].reshape(s_size, h_size))
            fc1_b = torch.from_numpy(weights[s_size*h_size:fc1_end])
            fc2_W = torch.from_numpy(weights[fc1_end:fc1_end+(h_size*a_size)].reshape(h_size, a_size))
            fc2_b = torch.from_numpy(weights[fc1_end+(h_size*a_size):])
            
            # set the weights for each layer
            self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
            self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
            self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
            self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))
        
        def get_weights_dim(self):
            """ Get weight dimension of neural network, Input -- Hidden -- Output --> 2 weight sets,
                - set_1 = number of state size input nodes (2 + 1) times number of hidden layer nodes (16)
                - set_2 = number of hidden layer nodes (16 + 1) times number of actions (1)
            
                INPUTS:
                ------------
                    None
                
                OUTPUTS:
                ------------
                    weights_dim - (int) number of single weights needed for fully connected neural network --> 65 values
            """
            weights_dim = (self.s_size+1)*self.h_size + (self.h_size+1)*self.a_size
            
            return weights_dim
            
        def forward(self, x):
            """ Forward pass of neural network 
                
                INPUTS:
                ------------
                    x - (torch tensor) two float values (two state variables), e.g. tensor([-0.4593,  0.0069])
                
                OUTPUTS:
                ------------
                    x - (torch tensor) one float value as action (between -1 and 1), e.g. tensor([0.9593])
            """
            x = F.relu(self.fc1(x))
            x = F.tanh(self.fc2(x))
            
            return x.cpu().data
            
        def evaluate(self, weights, gamma=1.0, max_t=5000):
            """ Evaluate episode
                
                INPUTS:
                ------------
                    weights - (numpy array) one dimensional, 65 values
                
                OUTPUTS:
                ------------
                    episode_return - (float) the return for the episode
            """
            self.set_weights(weights)
            episode_return = 0.0
            state = self.env.reset()
            for t in range(max_t):
                state = torch.from_numpy(state).float().to(device)
                action = self.forward(state)
                state, reward, done, _ = self.env.step(action)
                episode_return += reward * math.pow(gamma, t)
                if done:
                    break
        
            return episode_return
        
    agent = Agent(env).to(device)

    RESULTS:
    ------------
    observation space: Box(2,)
    action space: Box(1,)
     - low: [-1.]
     - high: [ 1.]
    ```
    ### Train the Agent with a Cross-Entropy Method
    ```
    def cem(n_iterations=500, max_t=1000, gamma=1.0, print_every=10, pop_size=50, elite_frac=0.2, sigma=0.5):
        """ PyTorch implementation of a cross-entropy method.
            
            INPUTS:
            ------------
                n_iterations - (int) maximum number of training iterations
                max_t - (int) maximum number of timesteps per episode
                gamma - (float) discount rate
                print_every - (int) how often to print average score (over last 100 episodes)
                pop_size - (int) size of population at each iteration
                elite_frac - (float) percentage of top performers to use in update
                sigma - (float) standard deviation of additive noise
                
            OUTPUTS:
            ------------
                scores - (list) accumulated rewards during episodes
        """
        n_elite=int(pop_size*elite_frac)

        scores_deque = deque(maxlen=100)
        scores = []
        best_weight = sigma*np.random.randn(agent.get_weights_dim())

        for i_iteration in range(1, n_iterations+1):
            weights_pop = [best_weight + (sigma*np.random.randn(agent.get_weights_dim())) for i in range(pop_size)]
            rewards = np.array([agent.evaluate(weights, gamma, max_t) for weights in weights_pop])

            elite_idxs = rewards.argsort()[-n_elite:]
            elite_weights = [weights_pop[i] for i in elite_idxs]
            best_weight = np.array(elite_weights).mean(axis=0)

            reward = agent.evaluate(best_weight, gamma=1.0)
            scores_deque.append(reward)
            scores.append(reward)
            
            torch.save(agent.state_dict(), 'checkpoint.pth')
            
            if i_iteration % print_every == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))

            if np.mean(scores_deque)>=90.0:
                print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration-100, np.mean(scores_deque)))
                break
        return scores

    scores = cem()

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    ```
    ### Watch a Smart Agent!
    ```
    # load the weights from file
    agent.load_state_dict(torch.load('checkpoint.pth'))

    state = env.reset()
    img = plt.imshow(env.render(mode='rgb_array'))
    while True:
        state = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            action = agent(state)
        img.set_data(env.render(mode='rgb_array')) 
        plt.axis('off')
        display.display(plt.gcf())
        display.clear_output(wait=True)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if done:
            break

    env.close()
    ```
    ![image13]

    ![image14]

## Setup Instructions <a name="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a name="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project <a name="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/Sparkify-Project.git
```

- Change Directory
```
$ cd Sparkify-Project
```

- Create a new Python environment, e.g. spark_env. Inside Git Bash (Terminal) write:
```
$ conda create --name spark_env
```

- Activate the installed environment via
```
$ conda activate spark_env
```

- Install the following packages (via pip or conda)
```
numpy = 1.12.1
pandas = 0.23.3
matplotlib = 2.1.0
seaborn = 0.8.1
pyspark = 2.4.3
```

- Check the environment installation via
```
$ conda env list
```

## Acknowledgments <a name="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Data Science'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a name="Further_Links"></a>

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

Further Deep Reinforcement Learning References
* [Very good summary of DQN](https://medium.com/@nisheed/udacity-deep-reinforcement-learning-project-1-navigation-d16b43793af5)
* [Cheatsheet](https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/cheatsheet/cheatsheet.pdf)
* [Reinforcement Learning Textbook](https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf)
* [Reinforcement Learning Textbook - GitHub Repo to Python Examples](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
* [Udacity DRL Github Repository](https://github.com/udacity/deep-reinforcement-learning)
* [Open AI Gym - Installation Guide](https://github.com/openai/gym#installation)
* [Deep Reinforcement Learning Nanodegree Links](https://docs.google.com/spreadsheets/d/19jUvEO82qt3itGP3mXRmaoMbVOyE6bLOp5_QwqITzaM/edit#gid=0)
