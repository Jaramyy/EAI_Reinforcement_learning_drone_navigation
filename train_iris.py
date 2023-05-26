import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.omniverse_isaacgym_utils import get_env_instance
from skrl.envs.torch import wrap_env
from skrl.utils import set_seed


# set the seed for reproducibility
set_seed(42)


# Define the models (stochastic and deterministic models) for the agent using helper mixin.
# - Policy: takes as input the environment's observation/state and returns an action
# - Value: takes the state as input and provides a value to guide the policy
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU(),
                                 nn.Linear(64, 32),
                                 nn.ELU(),
                                 nn.Linear(32, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
        # self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
        #                     nn.ELU(),
        #                     nn.Linear(64, 32),
        #                     nn.ELU(),
        #                     nn.Linear(32, self.num_actions))
        # self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))    

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}
class PolicyRNN(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2,
                 num_envs=1, num_layers=1, hidden_size=400, sequence_length=20):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hout
        self.sequence_length = sequence_length

        self.rnn = nn.RNN(input_size=self.num_observations,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)  # batch_first -> (batch, sequence, features)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
                            nn.ELU(),
                            nn.Linear(64, 32),
                            nn.ELU(),
                            nn.Linear(32, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))    

    def compute(self, inputs, role):
        # training
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]

        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            # get the hidden states corresponding to the initial sequence
            sequence_index = 1 if role == "target_policy" else 0  # target networks act on the next state of the environment
            hidden_states = hidden_states[:,:,sequence_index,:].contiguous()  # (D * num_layers, N, Hout)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, hidden_states = self.rnn(rnn_input[:,i0:i1,:], hidden_states)
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)
        # rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        return self.net(rnn_output), self.log_std_parameter, {}
    
class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU(),
                                 nn.Linear(64, 32),
                                 nn.ELU(),
                                 nn.Linear(32, 1))
        # self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
        #                     nn.ELU(),
        #                     nn.Linear(64, 32),
        #                     nn.ELU(),
        #                     nn.Linear(32, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}
class ValueRNN(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 num_envs=1, num_layers=1, hidden_size=400, sequence_length=20):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hout
        self.sequence_length = sequence_length

        self.rnn = nn.RNN(input_size=self.num_observations,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)  # batch_first -> (batch, sequence, features)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
                            nn.ELU(),
                            nn.Linear(64, 32),
                            nn.ELU(),
                            nn.Linear(32, 1))

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]

        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            # get the hidden states corresponding to the initial sequence
            sequence_index = 1 if role == "target_policy" else 0  # target networks act on the next state of the environment
            hidden_states = hidden_states[:,:,sequence_index,:].contiguous()  # (D * num_layers, N, Hout)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, hidden_states = self.rnn(rnn_input[:,i0:i1,:], hidden_states)
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)
        # rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        return self.net(rnn_output), {}


# define the model
class MLP(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        
        num_ramdom_observations = 4 
        self.fc1 = nn.Linear(num_ramdom_observations, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1 ) #self.num_actions

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs):
        x = self.fc1(inputs)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return torch.relu(x), self.log_std_parameter, {}
    


# instance VecEnvBase and setup task
headless = False  # set headless to False for rendering
env = get_env_instance(headless=headless)



from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
# from crazyflie import CrazyflieTask2, TASK_CFG
# from iris import irisTask, TASK_CFG

if headless is True:
    from omni.isaac.core.utils.extensions import enable_extension
    enable_extension("omni.replicator.isaac")  # required by OIGE

from iris_random_physical import irisTask, TASK_CFG

TASK_CFG["headless"] = headless
# TASK_CFG["task"]["env"]["numEnvs"] = 1024
# TASK_CFG["task"]["env"]["controlSpace"] = "cartesian"  # "joint" or "cartesian"

sim_config = SimConfig(TASK_CFG)



# task = CrazyflieTask2(name="ReachingFranka", sim_config=sim_config, env=env)
task = irisTask(name="ReachingFranka", sim_config=sim_config, env=env)
env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=True)

# wrap the environment
env = wrap_env(env, "omniverse-isaacgym")

device = env.device


# Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)


# Instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#spaces-and-models
print(env.observation_space)

models_ppo = {}
models_ppo["policy"] = Policy(env.observation_space, env.action_space, device)
models_ppo["value"] = Value(env.observation_space, env.action_space, device)
models_ppo["mlp"] = MLP(env.observation_space, env.action_space, device)

# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#configuration-and-hyperparameters
cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["rollouts"] = 16 #16
cfg_ppo["learning_epochs"] = 8
cfg_ppo["mini_batches"] = 32
cfg_ppo["discount_factor"] = 0.99
cfg_ppo["lambda"] = 0.95
cfg_ppo["learning_rate"] = 1e-4
cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL
cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.016}
cfg_ppo["random_timesteps"] = 0
cfg_ppo["learning_starts"] = 0
cfg_ppo["grad_norm_clip"] = 1.0
cfg_ppo["ratio_clip"] = 0.2
cfg_ppo["value_clip"] = 0.2
cfg_ppo["clip_predicted_values"] = True
cfg_ppo["entropy_loss_scale"] = 0.02     #0.02 Increase 
cfg_ppo["value_loss_scale"] = 2.0
cfg_ppo["kl_threshold"] = 0.016
cfg_ppo["state_preprocessor"] = RunningStandardScaler
cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg_ppo["value_preprocessor"] = RunningStandardScaler
cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints each 32 and 250 timesteps respectively
cfg_ppo["experiment"]["write_interval"] = 32
cfg_ppo["experiment"]["checkpoint_interval"] = 250



agent = PPO(models=models_ppo,
            memory=memory,
            cfg=cfg_ppo,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 500000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()