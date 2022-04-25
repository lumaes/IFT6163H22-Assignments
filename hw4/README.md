# Re-run experiment

**DATA folder was to big to upload on gradescope, if you want them please send me a private message on discord @Lucas M**

Here a the list of commands used to run experiments. Make sure that you are running the conda env named roble and that all the requirements.txt has been installed (check default github repository for that).


**rem:** Code for generating the plots if a Jupyter notebook in ` ift6163/` named `ift6163_hw4_plots.ipynb`


## Default config file
##### (modified from default repo)

```yaml
env_name: 'LunarLander-v3'
atari: True
ep_len: 200
exp_name: 'todo'
double_q: False
batch_size: 64
train_batch_size: 64
eval_batch_size: 4096
num_agent_train_steps_per_iter: 2
num_critic_updates_per_agent_update: 2
seed: 1
no_gpu: False
which_gpu: 0
video_log_freq: -1
scalar_log_freq: 1
save_params: False
rl_alg: 'dqn'
learning_starts: 1024
learning_freq: 0
target_update_freq: 1
exploration_schedule: 0
optimizer_spec:  0
replay_buffer_size: 1000000
frame_history_len: 1
gamma: 0.99
n_layers_critic: 2
size_hidden_critic: 64
critic_learning_rate: 1e-3
n_layers: 2
size: 64
learning_rate: 1e-3
ob_dim: 0
ac_dim: 0
discrete: True
grad_norm_clipping: True
n_iter: 10000
polyak_avg: 0.01 ##
# ===================== #
# TD3 CONFIG PARAMETERS #
# ===================== #
td3_target_policy_noise: 0.1 ##
td3_noise_clip: 0.5
td3_two_q_net: False
td3_policy_delay: 2
```

## Experiment 1

```bash
# command 1
python run_hw4.py env_name=MsPacman-v0 exp_name=q1 n_iter=1000000 \
num_agent_train_steps_per_iter=1
```

## Experiment 2

```bash
# command 1
python3 run_hw4.py env_name=LunarLander-v3 exp_name=q2_dqn_1 seed=1 n_iter=350000 \
num_agent_train_steps_per_iter=1
```

```bash
# command 2
python3 run_hw4.py env_name=LunarLander-v3 exp_name=q2_dqn_2 seed=2 n_iter=350000 \
num_agent_train_steps_per_iter=1
```

```bash
# command 3
python3 run_hw4.py env_name=LunarLander-v3 exp_name=q2_dqn_3 seed=3 n_iter=350000 \
num_agent_train_steps_per_iter=1
```

```bash
# command 4
python3 run_hw4.py env_name=LunarLander-v3 exp_name=q2_doubledqn_1 double_q=True \
seed=1 n_iter=350000 num_agent_train_steps_per_iter=1
```

```bash
# command 5
python3 run_hw4.py env_name=LunarLander-v3 exp_name=q2_doubledqn_2 double_q=True \
seed=2 n_iter=350000 num_agent_train_steps_per_iter=1
```

```bash
# command 6
python3 run_hw4.py env_name=LunarLander-v3 exp_name=q2_doubledqn_3 double_q=True \
seed=3 n_iter=350000 num_agent_train_steps_per_iter=1
```

## Experiment 3

```bash
# command 1
python run_hw4.py env_name=LunarLander-v3 exp_name=q3_default \
n_iter=350000 num_agent_train_steps_per_iter=1
```

```bash
# command 2
python run_hw4.py env_name=LunarLander-v3 exp_name=q3_layer1 \
n_iter=350000 num_agent_train_steps_per_iter=1 n_layers_critic=1
```

```bash
# command 3
python run_hw4.py env_name=LunarLander-v3 exp_name=q3_layer5 \
n_iter=350000 num_agent_train_steps_per_iter=1 n_layers_critic=5
```

```bash
# command 4
python run_hw4.py env_name=LunarLander-v3 exp_name=q3_layer8 \
n_iter=350000 num_agent_train_steps_per_iter=1 n_layers_critic=8
```

## Experiment 4

```bash
# command 1
python3 run_hw4.py exp_name=q4_ddpg_up1_1_lr1e-4 rl_alg=ddpg \
env_name=InvertedPendulum-v2 atari=False learning_freq=1 \
num_agent_train_steps_per_iter=1 num_critic_updates_per_agent_update=1 \
n_iter=350000 learning_rate=1e-4
```

```bash
# command 2
python3 run_hw4.py exp_name=q4_ddpg_up1_1_lr1e-3rl_alg=ddpg \
env_name=InvertedPendulum-v2 atari=False learning_freq=1 \
num_agent_train_steps_per_iter=1 num_critic_updates_per_agent_update=1 \
n_iter=350000 learning_rate=1e-3
```

```bash
# command 3
python3 run_hw4.py exp_name=q4_ddpg_up1_1_lr1e-5 rl_alg=ddpg \
env_name=InvertedPendulum-v2 atari=False learning_freq=1 \
num_agent_train_steps_per_iter=1 num_critic_updates_per_agent_update=1 \
n_iter=350000 learning_rate=1e-5
```

```bash
# command 4
python3 run_hw4.py exp_name=q4_ddpg_up1_3_lr1e-5 rl_alg=ddpg \
env_name=InvertedPendulum-v2 atari=False learning_freq=1 \
num_agent_train_steps_per_iter=3 \
num_critic_updates_per_agent_update=1 n_iter=350000 learning_rate=1e-5
```

```bash
# command 5
python3 run_hw4.py exp_name=q4_ddpg_up3_1_lr1e-5  rl_alg=ddpg \
env_name=InvertedPendulum-v2 atari=False learning_freq=1 \
num_agent_train_steps_per_iter=1 \
num_critic_updates_per_agent_update=3 n_iter=350000 learning_rate=1e-5
```

```bash
# command 6
python3 run_hw4.py exp_name=q4_ddpg_up3_3_lr1e-5 rl_alg=ddpg \
env_name=InvertedPendulum-v2 atari=False learning_freq=1 \
num_agent_train_steps_per_iter=3 \
num_critic_updates_per_agent_update=3 n_iter=350000 learning_rate=1e-5
```

## Experiment 5

```bash
# command 1
python run_hw4.py exp_name=q5_ddpg_hard_up3_3_lr1-e5 rl_alg=ddpg \
env_name=HalfCheetah-v2 atari=false learning_freq=1 \
num_agent_train_steps_per_iter=3 num_critic_updates_per_agent_update=3 \
n_iter=333333 learning_rate=1e-5
```

### Second Experiment

```bash
# command 2
python3 run_hw4.py exp_name=q5_ddpg_hard_up_lr1-e5 rl_alg=ddpg \
env_name=HalfCheetah-v2 atari=false learning_freq=1 \
num_agent_train_steps_per_iter=1 num_critic_updates_per_agent_update=1 \
n_iter=1000000 learning_rate=1e-5 critic_learning_rate=1e-4
```

## Experiment 6

```bash
# command 1
python run_hw4.py exp_name=q6_td3_shape_default_rho_default \
rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false learning_freq=1 \
num_agent_train_steps_per_iter=1 num_critic_updates_per_agent_update=1 \
n_iter=350000
```

```bash
# command 2
python run_hw4.py exp_name=q6_td3_shape_default_rho_0.01 rl_alg=ddpg \
env_name=InvertedPendulum-v2 atari=false learning_freq=1 \
num_agent_train_steps_per_iter=1 num_critic_updates_per_agent_update=1 \
n_iter=350000 td3_target_policy_noise=0.01
```

```bash
# command 3
python run_hw4.py exp_name=q6_td3_shape2l128hu_rho0.2 rl_alg=ddpg \
env_name=InvertedPendulum-v2 atari=false learning_freq=1 \
num_agent_train_steps_per_iter=1 num_critic_updates_per_agent_update=1 \
n_iter=350000 td3_target_policy_noise=0.2
```

```bash
# command 4
python run_hw4.py exp_name=q6_td3_shape_default_rho0.5 rl_alg=ddpg \
env_name=InvertedPendulum-v2 atari=false learning_freq=1 \
num_agent_train_steps_per_iter=1 num_critic_updates_per_agent_update=1 \
n_iter=350000 td3_target_policy_noise=0.5
```

```bash
# command 5
python run_hw4.py exp_name=q6_td3_shape_2l_64u_rho0.2 rl_alg=ddpg \
env_name=InvertedPendulum-v2 atari=false learning_freq=1 \
num_agent_train_steps_per_iter=1 num_critic_updates_per_agent_update=1 \
n_iter=350000 td3_target_policy_noise=0.2 n_layers_critic=2 size_hidden_critic=64
```

```bash
# command 6
python run_hw4.py exp_name=q6_td3_shape_1l_128u_rho0.2 rl_alg=ddpg \
env_name=InvertedPendulum-v2 atari=false learning_freq=1 \
num_agent_train_steps_per_iter=1\
num_critic_updates_per_agent_update=1 n_iter=350000 td3_target_policy_noise=0.2 \
n_layers_critic=1 size_hidden_critic=128
```

```bash
# command 7
python run_hw4.py exp_name=q6_td3_shape_4l_32u_rho0.2 rl_alg=ddpg \
env_name=InvertedPendulum-v2 atari=false learning_freq=1 \
num_agent_train_steps_per_iter=1 num_critic_updates_per_agent_update=1 n_iter=350000 \
td3_target_policy_noise=0.2 n_layers_critic=4 size_hidden_critic=32
```

## Experiment 6 : BONUS
see code.

## Experiment 7

```bash
python run_hw4.py exp_name=q7_td3_shape_default_u_3_3_rho0.5 rl_alg=td3 \
env_name=HalfCheetah-v2 atari=false  num_agent_train_steps_per_iter=3 \
num_critic_updates_per_agent_update=3  learning_freq=1 n_iter=333333 \
td3_target_policy_noise=0.5 learning_rate=1e-5 critic_learning_rate=1e-4
```
