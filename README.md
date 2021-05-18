[image1]: assets/intro_policy_based_methods.png "image1"


# Deep Reinforcement Learning Theory - Policy Based Methods

## Content 
- [Introduction](#intro)
- [Policy Based Methods](#policy_based_methods)
    - [Policy Function Approximation](#Policy_Function_Approximation)
    - [More on the Policy](#More_on_the_Policy)
    - [Hill Climbing](#hill_climbing)
    - [Hill Climbing Pseudocode](hill_climbing_pseudo)
    - [Beyond Hill Climbing](#beyond_hill_climbing)
    - [More Black-Box Optimization](#black_box)
    - [Why Policy Based Methods?](#why_policy_based_methods)

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

## Policy Function Approximation <a name="Policy_Function_Approximation"></a> 
- In deep reinforcement learning, it is common to represent the policy with a neural network.
    - This network takes the environment state as **input**.
    - If the environment has discrete actions, the **output** layer has a node for each possible action and contains the probability that the agent should select each possible action.
- The weights in this neural network are initially set to random values. Then, the agent updates the weights as it interacts with (and learns more about) the environment.


## More on the Policy <a name="More_on_the_Policy"></a> 
- Policy-based methods can learn either stochastic or deterministic policies, and they can be used to solve environments with either finite or continuous action spaces. 

## Hill Climbing <a name="hill_climbing"></a> 
- Hill climbing is an iterative algorithm that can be used to find the weights &theta;  for an optimal policy.
- At each iteration,
    - We slightly perturb the values of the current best estimate for the weights &theta;<sub>best</sub>, to yield a new set of weights.
    - These new weights are then used to collect an episode. If the new weights &theta;<sub>new</sub> resulted in higher return than the old weights, then we set &theta;<sub>best</sub> ← &theta;<sub>new</sub>.
        
## Hill Climbing Pseudocode <a name="hill_climbing_pseudo"></a> 

## Beyond Hill Climbing <a name="eyond_hill_climbing"></a> 
- **Steepest ascent hill climbing** is a variation of hill climbing that chooses a small number of neighboring policies at each iteration and chooses the best among them.
- **Simulated annealing** uses a pre-defined schedule to control how the policy space is explored, and gradually reduces the search radius as we get closer to the optimal solution.
- **Adaptive noise scaling** decreases the search radius with each iteration when a new best policy is found, and otherwise increases the search radius.

## More Black-Box Optimization <a name="black_box"></a>
- The **cross-entropy method** iteratively suggests a small number of neighboring policies, and uses a small percentage of the best performing policies to calculate a new estimate.
- The **evolution strategies** technique considers the return corresponding to each candidate policy. 
- The policy estimate at the next iteration is a weighted sum of all of the candidate policies, where policies that got higher return are given higher weight.
 

## Why Policy Based Methods? <a name="why_policy_based_methods"></a>
- There are three reasons why we consider policy-based methods:
    1. **Simplicity**: Policy-based methods directly get to the problem at hand (estimating the optimal policy), without having to store a bunch of additional data (i.e., the action values) that may not be useful.
    2. **Stochastic policies**: Unlike value-based methods, policy-based methods can learn true stochastic policies.
    3. **Continuous action spaces**: Policy-based methods are well-suited for continuous action spaces.




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
