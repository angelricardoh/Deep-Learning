# CSCI-566 Assignment 3

## The objectives of this assignment
* Getting familiar with reinforcement learning and policy gradient methods!
* Setting up and interacting with environments, specifically [OpenAI Gym environments](https://github.com/openai/gym).
* Implementing REINFORCE: rollout storage, policy network and training loop.
* Testing RL on various environments and engineering reward function
* Implementing Actor Critic Architecture


## Work on the assignment

### Colab
You can use Google Colab, but training should not be any faster than a usual machine since we won't be using GPUs.

To do this, simply click on the "Open in Colab" button at the top of each notebook while viewing the notebook on Github (or locally). Make sure to right click -> open in new tab for it to work on Github.

You can then make a copy of the notebook in Google Colab (Copy to Drive) and start working on it!

Make sure when you download your copy of .ipynb notebook to submit, it has the outputs plots and videos.
It could happen that videos do not appear on local jupyter notebook if you download it from Colab. But the videos should still be stored there, and can be checked by converting it to html.

`jupyter nbconvert --to html Policy_Gradients.ipynb`

You can check it locally (but do not submit the generated .html file)

----
### Your computer

Please first clone or download as .zip file of this repository.

Working on the assignment in a virtual environment is highly encouraged.

In this assignment, we recommend you use Python `3.6.9` (`3.6.` or `3.7` should work too).
You will need to make sure that your virtualenv setup is of the correct version of python.
We will be using *PyTorch* in this assignment.

Please see below for executing a virtual environment.
```shell
cd CSCI566-Assignment3
pip3 install virtualenv # If you didn't install it
virtualenv -p $(which python3) ./venv_cs566_hw3
source ./venv_cs566_hw3/bin/activate

# Install dependencies
pip3 install -r requirements.txt

# Work on the assignment

# Deactivate the virtual environment when you are done
deactivate
```

## Work with IPython Notebook
To start working on the assignment, simply run the following command to start an ipython kernel.
```shell
# add your virtual environment to jupyter notebook
python -m ipykernel install --user --name=venv_cs566_hw3

# port is only needed if you want to work on more than one notebooks
jupyter notebook --port=<your_port>

```
and then work on the problem in `Policy_Gradients.ipynb` notebooks.
Check the python environment you are using on the top right corner.
If the name of environment doesn't match, change it to your virtual environment in "Kernel>Change kernel".

## Working on the Problem
In the notebook file `Policy_Gradients.ipynb`, we indicate `TODO` or `Your Code` for you to fill in with your implementation.
You only need to edit this notebook, and only inside the specified TODO blocks.

## PLEASE DO NOT CLEAR THE OUTPUT OF THE CELLS IN THE .ipynb FILES
Your outputs on the `.ipynb` files will be graded. We will not rerun the code. If the outputs are missing, that will will be considered as if it is not attempted, and such cases will be penalized.

## How to submit

Run the following command to zip all the necessary files for submitting your assignment.

```shell
sh collectSubmission.sh <USC_ID>
```

This will create a file named `<USC_ID>.zip` (eg. 4916525888.zip). Please submit this file through the [Google form](https://forms.gle/MESM9xkbje2KVfje7).
If you have to create own .zip file, make sure to ONLY include the `Policy_Gradients.ipynb` file, and name your file as `<USC_ID>.zip`.

We will deduct points if you don't follow the above submission guideline.

## Questions?
If you have any question or find a bug in this assignment (or even any suggestions), we are
more than welcome to assist. Please take a look at the FAQ section below before posting a question.

Again, NO INDIVIDUAL EMAILS WILL BE RESPONDED.

PLEASE USE **PIAZZA** TO POST QUESTIONS (under folder hw3).


## FAQ

- **Can I reuse the virtualenv from previous assignments?**  
You can reuse the virtual environment but you should install the missing packages using `pip3 install -r requirements.txt`.  
Usually it is simpler to create a new virtualenv, as given in the instructions above.

- **Do I need to retain training outputs like videos in the ipython notebook?**  
**Yes!** Please do not manually clear any outputs from your notebooks (Note that sometimes the given code will clear the output for you, but you do not need to worry about it).

- **Section 1: What's a wrapper?**  
Wrappers are used to add some functionality over existing classes. Carefully look at the wrapper's   __init__  function to see what it expects as input and how you can use it.

- **Section 3: What is entropy loss?**  
Read the description given in the assignment. You can also refer to the lecture 19 slides on RL in practice to read more about it.

- **Section 4 (First thing you train): My reward curve goes up for a while, but suddenly drops and stops learning. Is it a problem of hyperparameters?**  
No, your implementation in    def train(...)  function has a problem. Look at TODO block #3 there to figure it out.

- **My rollout time is 3-4 seconds. Is there something wrong with my code?**  
On our machines, the rollout time was <2 seconds, but it could be that your machine is slower so it takes longer. Just make sure it is constant, and doesn't increase linearly.

- **Reward design experiment: Dense reward environment performs better than the engineered reward. Is it right?**  
If everything seems to learn well, it should be right. You are expected to write your observations and reasoning behind them in the subjective question. We will be lenient about these questions.

- **The reward curves can vary a lot. How will you grade the assignments?**  
We will look at both, the reward curves as well as the lines of code implemented by you while grading. We are not planning to re-run any code, so make sure you retain all the outputs.

- **Actor Critic network with baseline seems to be very unstable. What am I doing wrong?**  
It is very likely that your actor critic network is unstable, and learns poorly. But it should show signs of learning, and not be completely flat. We will grade this part mostly based on the code, and your explanations in the questions below. HINT: compare it with REINFORCE and PPO.

- **In the actor critic code, what does detach function do? Do I need to use it?**  
You can read detach documentation here: https://pytorch.org/docs/stable/autograd.html#torch.Tensor.detach
Basically, we detach variables when we do not want to propagate gradients through them. Hence, while computing advantage, we detach the value coming from the critic since we do not want to update the critic when we update the policy. We update critic separately through critic_loss. So we should not detach the critic's value outputs while computing critic loss.

- **I get strange errors when I simply copied over the code for Wall/Lava. What to do?**  
Read the instructions again. You are supposed to change a couple of things by understanding previous code. And then it should run fine.

- **Can I change hyperparameters for Wall/Lava environments? Can I change the environment code?**  
Feel free to change the hyperparameters in the final todo block. You can run it for however long you want (note that we won't run your code again).
Please don't change any environment code that we have provided.
