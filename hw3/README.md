# Re-run experiment



Here a the list of commands used to run experiments. Make sure that you are running the conda env named roble and that all the requirements.txt has been installed (check default github repository for that).



**rem:** Code for generating the plots if a Jupyter notebook in ` ift6163/` named `ift6163_hw3_plots.ipynb`


## Default config file
##### (modified from default repo)

```yaml

env_name: 'CartPole-v0' # ['cheetah-ift6163-v0', 'reacher-ift6163-v0', 'obstacles-ift6163-v0' ]
ep_len: 200
exp_name: 'todo'
n_iter: 1
mpc_horizon: 10
mpc_num_action_sequences: 1000
mpc_action_sampling_strategy: 'random'
cem_iterations: 4
cem_num_elites: 5
cem_alpha: 1
add_sl_noise: True
batch_size_initial: 5000
batch_size: 8000
train_batch_size: 512
eval_batch_size: 400
seed: 1
no_gpu: False
which_gpu: 1
video_log_freq: -1
scalar_log_freq: 1
save_params: False
rl_alg: 'todo'

computation_graph_args:
   learning_rate: 0.001
   n_layers: 2
   size: 128
   ensemble_size: 3
   num_grad_steps_per_target_update: 1
   num_target_updates: 1
   
estimate_advantage_args:
   discount: 0.95
   gae_lambda: None
   standardize_advantages: False
   reward_to_go: False
   nn_baseline: False
   
train_args:
   num_agent_train_steps_per_iter: 1
   num_critic_updates_per_agent_update: 1
   num_actor_updates_per_agent_update: 1
   discrete: False
   ob_dim:  0
   ac_dim: 0
   ppo_clipping: False
   ppo_k_epochs: 20
   ppo_epsilon: 0.2
```


## Experiment 1

```bash
# command 1
python run_hw3.py env_name=CartPole-v0 n_iter=100 batch_size=1000 \
estimate_advantage_args.standardize_advantages=false \
exp_name=q1_sb_no_rtg_dsa rl_alg=reinforce

# command 2
python run_hw3.py env_name=CartPole-v0 n_iter=100 batch_size=1000 \
estimate_advantage_args.standardize_advantages=false \
estimate_advantage_args.reward_to_go=true exp_name=q1_sb_rtg_dsa \
rl_alg=reinforce

# command 3
python run_hw3.py env_name=CartPole-v0 n_iter=100 batch_size=1000\
estimate_advantage_args.reward_to_go=true exp_name=q1_sb_rtg_na\ rl_alg=reinforce

# command 4
python run_hw3.py env_name=CartPole-v0 n_iter=100 batch_size=5000 \
estimate_advantage_args.standardize_advantages=false \
exp_name=q1_lb_no_rtg_dsa rl_alg=reinforce

# command 5
python run_hw3.py env_name=CartPole-v0 n_iter=100 batch_size=5000 \
estimate_advantage_args.standardize_advantages=false \
estimate_advantage_args.reward_to_go=true exp_name=q1_lb_rtg_dsa \
rl_alg=reinforce

# command 6
python run_hw3.py env_name=CartPole-v0 n_iter=100 batch_size=5000 \
estimate_advantage_args.reward_to_go=true exp_name=q1_lb_rtg_na \
rl_alg=reinforce
```

## Experiment 2

```bash
# command 1
python run_hw3.py env_name=InvertedPendulum-v2 ep_len=1000 \
estimate_advantage_args.discount=0.9 n_iter=100 \
computation_graph_args.n_layers=2 computation_graph_args.size=64 \
batch_size=1000 computation_graph_args.learning_rate=0.01 \
estimate_advantage_args.reward_to_go=true exp_name=q2_b1000_r0.02 \
rl_alg=reinforce
```

## Experiment 3

```bash
# command 1
python run_hw3.py env_name=LunarLanderContinuous-v2 ep_len=1000 \
estimate_advantage_args.discount=0.99 n_iter=100 \
computation_graph_args.n_layers=2 computation_graph_args.size=64 \
batch_size=40000 train_batch_size=40000 \
computation_graph_args.learning_rate=0.005 \
estimate_advantage_args.reward_to_go=true \
estimate_advantage_args.nn_baseline=true rl_alg=reinforce \
exp_name=q3_b40000_r0.005
```

## Experiment 4

### Experiment 4.1

```bash
# command 1
python run_hw3.py env_name=HalfCheetah-v2 ep_len=150 \
estimate_advantage_args.discount=0.95 n_iter=100 \
computation_graph_args.n_layers=2 computation_graph_args.size=32 \
batch_size=10000 train_batch_size=10000 \
computation_graph_args.learning_rate=0.005 \
estimate_advantage_args.reward_to_go=true \
estimate_advantage_args.nn_baseline=true rl_alg=reinforce \
exp_name=q4_search_b10000_lr0.005_rtg_nnbaseline

# command 2
python run_hw3.py env_name=HalfCheetah-v2 ep_len=150 \
estimate_advantage_args.discount=0.95 n_iter=100 \
computation_graph_args.n_layers=2 computation_graph_args.size=32 \
batch_size=10000 train_batch_size=10000 \
computation_graph_args.learning_rate=0.01 \
estimate_advantage_args.reward_to_go=true \
estimate_advantage_args.nn_baseline=true rl_alg=reinforce \
exp_name=q4_search_b10000_lr0.01_rtg_nnbaseline

# command 3
python run_hw3.py env_name=HalfCheetah-v2 ep_len=150 \
estimate_advantage_args.discount=0.95 n_iter=100 \
computation_graph_args.n_layers=2 computation_graph_args.size=32 \
batch_size=10000 train_batch_size=10000 \
computation_graph_args.learning_rate=0.02 \
estimate_advantage_args.reward_to_go=true \
estimate_advantage_args.nn_baseline=true rl_alg=reinforce \
exp_name=q4_search_b10000_lr0.02_rtg_nnbaseline

# command 4
python run_hw3.py env_name=HalfCheetah-v2 ep_len=150 \
estimate_advantage_args.discount=0.95 n_iter=100 \
computation_graph_args.n_layers=2 computation_graph_args.size=32 \
batch_size=30000 train_batch_size=30000 \
computation_graph_args.learning_rate=0.005 \
estimate_advantage_args.reward_to_go=true \
estimate_advantage_args.nn_baseline=true rl_alg=reinforce \
exp_name=q4_search_b30000_lr0.005_rtg_nnbaseline

# command 5
python run_hw3.py env_name=HalfCheetah-v2 ep_len=150 \
estimate_advantage_args.discount=0.95 n_iter=100 \
computation_graph_args.n_layers=2 computation_graph_args.size=32 \
batch_size=30000 train_batch_size=30000 \
computation_graph_args.learning_rate=0.01 \
estimate_advantage_args.reward_to_go=true \
estimate_advantage_args.nn_baseline=true rl_alg=reinforce \
exp_name=q4_search_b30000_lr0.01_rtg_nnbaseline

# command 6
python run_hw3.py env_name=HalfCheetah-v2 ep_len=150 \
estimate_advantage_args.discount=0.95 n_iter=100 \
computation_graph_args.n_layers=2 computation_graph_args.size=32 \
batch_size=30000 train_batch_size=30000 \
computation_graph_args.learning_rate=0.02 \
estimate_advantage_args.reward_to_go=true \
estimate_advantage_args.nn_baseline=true rl_alg=reinforce \
exp_name=q4_search_b30000_lr0.02_rtg_nnbaseline

# command 7
python run_hw3.py env_name=HalfCheetah-v2 ep_len=150 \
estimate_advantage_args.discount=0.95 n_iter=100 \
computation_graph_args.n_layers=2 computation_graph_args.size=32 \
batch_size=50000 train_batch_size=50000 \
computation_graph_args.learning_rate=0.005 \
estimate_advantage_args.reward_to_go=true \
estimate_advantage_args.nn_baseline=true rl_alg=reinforce \
exp_name=q4_search_b50000_lr0.005_rtg_nnbaseline

# command 8
python run_hw3.py env_name=HalfCheetah-v2 ep_len=150 \
estimate_advantage_args.discount=0.95 n_iter=100 \
computation_graph_args.n_layers=2 computation_graph_args.size=32 \
batch_size=50000 train_batch_size=50000 \
computation_graph_args.learning_rate=0.01 \
estimate_advantage_args.reward_to_go=true \
estimate_advantage_args.nn_baseline=true rl_alg=reinforce \
exp_name=q4_search_b50000_lr0.01_rtg_nnbaseline

# command 9
python run_hw3.py env_name=HalfCheetah-v2 ep_len=150 \
estimate_advantage_args.discount=0.95 n_iter=100 \
computation_graph_args.n_layers=2 computation_graph_args.size=32 \
batch_size=50000 train_batch_size=50000 \
computation_graph_args.learning_rate=0.02 \
estimate_advantage_args.reward_to_go=true \
estimate_advantage_args.nn_baseline=true rl_alg=reinforce \
exp_name=q4_search_b50000_lr0.02_rtg_nnbaseline
```

### Experiment 4.2

```bash
# command 1
python run_hw3.py env_name=HalfCheetah-v2 ep_len=150 rl_alg=reinforce \
estimate_advantage_args.discount=0.95 n_iter=100 computation_graph_args.n_layers=2 \
computation_graph_args.size=32 batch_size=50000 train_batch_size=50000 \
computation_graph_args learning_rate=0.02 exp_name=q4_b50000_r0.02

# command 2
python run_hw3.py env_name=HalfCheetah-v2 ep_len=150 rl_alg=reinforce \
estimate_advantage_args.discount=0.95 n_iter=100 computation_graph_args.n_layers=2 \
computation_graph_args.size=32 batch_size=50000 train_batch_size=50000 \
computation_graph_args.learning_rate=0.02 estimate_advantage_args.reward_to_go=true \
exp_name=q4_b50000_r0.02_rtg

# command 3
python run_hw3.py env_name=HalfCheetah-v2 ep_len=150 rl_alg=reinforce \
estimate_advantage_args.discount=0.95 n_iter=100 computation_graph_args.n_layers=2 \
computation_graph_args.size=32 batch_size=50000 train_batch_size=50000 \
computation_graph_args.learning_rate=0.02 estimate_advantage_args.nn_baseline=true \
exp_name=q4_b50000_r0.02_nnbaseline

# command 4
python run_hw3.py env_name=HalfCheetah-v2 ep_len=150 rl_alg=reinforce \
estimate_advantage_args.discount=0.95 n_iter=100 computation_graph_args.n_layers=2 \
computation_graph_args.size=32 batch_size=50000 train_batch_size=50000 \
computation_graph_args.learning_rate=0.02 estimate_advantage_args.reward_to_go=true \
estimate_advantage_args.nn_baseline=true exp_name=q4_b50000_r0.02_rtg_nnbaseline
```

## Experiment 5

```bash
# command 1
python run_hw3.py env_name=Hopper-v2 ep_len=1000 rl_alg=reinforce \
estimate_advantage_args.discount=0.99 n_iter=300 computation_graph_args.n_layers=2 \
computation_graph_args.size=32 batch_size=2000 computation_graph_args.learning_rate=0.001 \
estimate_advantage_args.reward_to_go=true estimate_advantage_args.nn_baseline=true \
estimate_advantage_args.gae_lambda=0 exp_name=q5_b2000_r0.001_lambda0

# command 2
python run_hw3.py env_name=Hopper-v2 ep_len=1000 rl_alg=reinforce \
estimate_advantage_args.discount=0.99 n_iter=300 computation_graph_args.n_layers=2 \
computation_graph_args.size=32 batch_size=2000 computation_graph_args.learning_rate=0.001 \
estimate_advantage_args.reward_to_go=true estimate_advantage_args.nn_baseline=true \
estimate_advantage_args.gae_lambda=0.95 exp_name=q5_b2000_r0.001_lambda0.95

# command 3
python run_hw3.py env_name=Hopper-v2 ep_len=1000 rl_alg=reinforce \
estimate_advantage_args.discount=0.99 n_iter=300 computation_graph_args.n_layers=2 \
computation_graph_args.size=32 batch_size=2000 computation_graph_args.learning_rate=0.001 \
estimate_advantage_args.reward_to_go=true estimate_advantage_args.nn_baseline=true \
estimate_advantage_args.gae_lambda=0.99exp_name=q5_b2000_r0.001_lambda0.99

# command 4
python run_hw3.py env_name=Hopper-v2 ep_len=1000 rl_alg=reinforce \
estimate_advantage_args.discount=0.99 n_iter=300 computation_graph_args.n_layers=2 \
computation_graph_args.size=32 batch_size=2000 computation_graph_args.learning_rate=0.001 \
estimate_advantage_args.reward_to_go=true estimate_advantage_args.nn_baseline=true \
estimate_advantage_args.gae_lambda=1 exp_name=q5_b2000_r0.001_lambda1
```

## Experiment 6

```bash
# command 1
python run_hw3.py env_name=CartPole-v0 n_iter=100 rl_alg=ac \
batch_size=1000 train_batch_size=1000 exp_name=q6_100_1 \
computation_graph_args.num_target_updates=100 \
computation_graph_args.num_grad_steps_per_target_update=1

# command 2
python run_hw3.py env_name=CartPole-v0 n_iter=100 rl_alg=ac \
batch_size=1000 train_batch_size=1000 exp_name=q6_1_100 \
computation_graph_args.num_target_updates=1 \
computation_graph_args.num_grad_steps_per_target_update=100

# command 3
python run_hw3.py env_name=CartPole-v0 n_iter=100 rl_alg=ac \
batch_size=1000 train_batch_size=1000 exp_name=q6_10_10 \
computation_graph_args.num_target_updates=10 \
computation_graph_args.num_grad_steps_per_target_update=10
```

## Experiment 7

```bash
# command 1
python run_hw3.py env_name=InvertedPendulum-v2 rl_alg=ac ep_len=1000\
estimate_advantage_args.discount=0.95 n_iter=100\
computation_graph_args.n_layers=2 computation_graph_args.size=64\
batch_size=5000 computation_graph_args.learning_rate=0.01\
exp_name=q7_10_10\
computation_graph_args.num_target_updates=10\
computation_graph_args.num_grad_steps_per_target_update=10

# command 2
python run_hw3.py env_name=HalfCheetah-v2 rl_alg=ac ep_len=150\
estimate_advantage_args.discount=0.90 \
scalar_log_freq=1 n_iter=150 computation_graph_args.n_layers=2 \
computation_graph_args.size=32 batch_size=30000 \
train_batch_size=30000 eval_batch_size=1500 \
computation_graph_args.learning_rate=0.02 exp_name=q7_c_10_10 \
computation_graph_args.num_target_updates=10 \
computation_graph_args.num_grad_steps_per_target_update=10
```

## Experiment 8

```bash
# command 1
python run_hw3.py exp_name=q8_cheetah_n500_arch1x32 env_name=cheetah-ift6163-v0 \
estimate_advantage_args.discount=0.95 computation_graph_args.n_layers=2 computation_graph_args.size=32 \
computation_graph_args.learning_rate=0.01 scalar_log_freq=1 n_iter=100 batch_size=5000 \
train_batch_size=1024 rl_alg=dyna

# command 2
python run_hw3.py exp_name=q8_cheetah_n500_arch1x32_2 env_name=cheetah-ift6163-v0  \
estimate_advantage_args.discount=0.95 computation_graph_args.n_layers=2 computation_graph_args.size=32 \
computation_graph_args.learning_rate=0.01 scalar_log_freq=1 n_iter=100 batch_size=2000 \
train_batch_size=1024 rl_alg=dyna
```