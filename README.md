# RBM-GoL
This repository is basically me learning to use Restricted Boltzmann Machines.

Since they are generative models, and a classic and simple example is generating Ising configurations, here I try to generate Conway's Game of Life (GoL) configurations.

The idea is that, since the settings are very similar (a grid of 1s and 0s for GoL and 1s and -1s for Ising) the generalization is straightforward. The main new problem is what kind of phases can be distinguished in GoL.

## Activity
In order to distinguis phases I defined a value associated to a configuration that I will call _activity_, defined as:
$$a_i =\frac{\sum_{all\;cells} abs(state_{i+1} - state_i)}{grid\;size} $$
and looking at the graph of this value we can see there are three cases:
* $a_i = 0$ for all $i>t*$ (like _config 0_)
* $a_i \neq 0$ but const for all $i>t*$ (_config 8_)
* $a_i$ oscillates for all $i>t*$ (_config 7_)

![alt text](/images/act_intro.png)

We therefore identify three states:
* `dead state` : the grid is completely empty, this corresponds to 0 activity
* `stable state` : the grid has some pattern that does not evolve anymore or it evolves in such a way that the activity is const, here we get both 0 and const value for the activity
* `periodic state` : the pattern has periodic activity, meaning that the states of the grid repeat themselves in a cycle

Now the task is just to analyze the activity and check whether is 0, const or periodic, being careful in the 0 case to also check the grid, since it may be both the `dead` and the `stable` state.

In order to find the periodicity we can use autocorrelation, with a couple of expedients to avoif mistakes in the classification (these are better shown directly in the notebook `test_gol.ipynb`)

## RBM Implementation and Dataset
Nothing special in the implementation of of the RBM, I build it using PyTorch functions, in order to use already made optimization algorithms and, if needed, be GPU compatible.
In a RBM we have one visible layer and one hidden layer, the input grid is flattened and used as input for the visible layer. Then we iteratively update the two layers. After $k$ iterations (where $k$ is a hyperparameter), we correct the weights based on how much the ending visible layer state differs from the input state.

After the training is done, we proceed with the sampling: this is the same process as the training, except we will not touch the weights this time. Basically for each input state in the test set, we sample a generated state from the RBM, ideally belonging to the right class of states.

In order to build a dataset I sampled random configurations, evolved them for a given number of epochs and then classified the initial state based on the activity. The function `create_set` allows both to have an uneven dataset, setting just the size, or a even distributed one, where the final dataset will be equally divided in three groups of `stable`, `dead` and `periodic` configurations.

## Results
I only tried with some simple RBM for now, in particular with 1 `hidden neuron` and the size of the grid as `visible neurons`. I trained an RBM for each case: the idea is that the trained model will allow to sample configurations that will produce a specific result, instead of giving random results.

The obtained results are in the tables below and do not show great performances: the _random_ column shows the percentage related to each category when sampling random initial configurations. The other three columns show the same percentages, but when sampling from an RBM trained on a specific configurations (`Dead_RBM` is the RBM trained on `dead` configurations).

As it is clearly shown, this basic RBM do not show any improvement with respect to the random sampling.

### 4x4 Grid
|        | Random   | Dead_RBM| Stable_RBM  | Periodic_RBM |
|:---:   | :------: |:-------:| :----------:| :----------: |
|Dead    | 75.47%   | 78.17%  | 71%         | 69.10%       |
|Stable  | 16.23%   | 15.57%  | 19.07%      | 20.60%       |
|Periodic| 8.3%     | 6.27%   | 9.93%       | 10.30%       |
### 5x5 Grid
|        | Random| Dead_RBM| Stable_RBM  | Periodic_RBM |
|:------:| :---: |:-------:| :----------:| :----------: |
|Dead    | 68.2% | 70.27%  | 69%         | 68.17%       |
|Stable  | 29%   | 27.83%  | 28.23%      | 29.6%        |
|Periodic| 2.8%  | 1.9%    | 2.77%       | 2.23%        |

## TO DO
* Analyze initial states of different classes
* Try to speed up the dataset creation
* Try bigger RBM and longer training
