import pickle as pk 
from collections import deque

from modnet.preprocessing import MODData
from modnet.models import MODNetModel

import gymnasium as gym
import gymnasium.spaces as spaces
import jax.numpy as jnp

import optax
from jax import random, jit, value_and_grad
from flax import linen as nn

from tqdm import tqdm

N_FEATS = 974

@jit
def mse_loss(y_true, y_pred):
    return jnp.mean(jnp.square(y_pred-y_true))

class DRLModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)  # Output a single Q-value for the given state-action pair
        
        return x

class ReplayBuffer:
    def __init__(self, batch_size):
        self.batch_size = batch_size 
        self.buffer = []
    
    def add(self, experiences):
        for experience in experiences:
            self.buffer.append(experience)
    
    def sample(self, key):
        choices = random.choice(key, 
                                a=len(self.buffer), 
                                shape=(min(self.batch_size, len(self.buffer)), ), 
                                replace=False)
        return [self.buffer[i] for i in choices]
    
    def __len__(self):
        return len(self.buffer)

class ModNetSearchEnv(gym.Env):
    def __init__(self, dataset, starting_inds, max_steps=10):
        super(ModNetSearchEnv, self).__init__()

        self.all_data = dataset
        self.all_actions = jnp.arange(len(data.get_featurized_df()))
        self.initial_inds = starting_inds
        self.initial_data = self.all_data.from_indices(starting_inds)
        
        self.candidate_inds = [x for x in range(len(self.all_data.get_featurized_df())) if x not in starting_inds]
        
        self.observed_inds = starting_inds

        self.max_steps = max_steps
        
    def reset(self):
        self.current_step = 0
        self.target_idx = None
        self.done = False
        
        self.best_reward = None
        self.best_action = None

        self.candidate_inds = self.all_actions[~jnp.isin(jnp.arange(len(self.all_data.get_featurized_df())), self.initial_inds)]
        self.action_space = spaces.Discrete(len(self.candidate_inds))

        self.observed_inds = self.initial_inds
        
        return self.initial_data

    def step(self, actions):
        if self.done:
            raise ValueError("Episode is done. Please call reset().")
        
        # Take action
        inds = self.candidate_inds[actions]
        observations = self.all_data.from_indices(inds)
        
        # Evaluate the vectors using lookup data
        true_rewards = jnp.array(self.all_data.targets[inds].flatten())
        
        # Update best true_rewards and action
        if self.best_reward is None or jnp.max(true_rewards) > self.best_reward:
            self.best_reward = jnp.max(true_rewards)
            self.best_action = self.candidate_inds[actions[jnp.argmax(true_rewards)]]
        
        # Update episode termination conditions
        self.current_step += 1

        if self.current_step >= self.max_steps:
            self.done = True

        # Update the observed and candidate search spaces
        self.observed_inds = jnp.hstack((self.observed_inds, inds))
        # Reduce the candidate pool
        self.candidate_inds = self.all_actions[~jnp.isin(self.all_actions, self.observed_inds)]
    
        if self.current_step < len(self.candidate_inds):
            self.target_idx = self.current_step
        else:
            self.target_idx = None
            self.done = True
        
        self.action_space = spaces.Discrete(len(self.candidate_inds))

        return observations, true_rewards, self.done, {"target_idx": self.target_idx, "best_reward": self.best_reward, "best_action": self.best_action}
    
class VectorSearchEnv(gym.Env):
    def __init__(self, X, y, observed_inds, max_steps=10):
        super(VectorSearchEnv, self).__init__()

        self.all_vectors = X
        self.all_targets = y
        self.all_actions = jnp.arange(len(X))

        self.initial_data = self.all_vectors[observed_inds, :]
        self.initial_targets = self.all_targets[observed_inds]
        self.initial_inds = observed_inds

        self.candidate_inds = self.all_actions[~jnp.isin(self.all_actions, self.initial_inds)]
        self.candidates = self.all_vectors[self.candidate_inds, :]
        self.candidate_targets = self.all_targets[self.candidate_inds]
        
        self.observed_inds = observed_inds
        self.observed_vectors = self.all_vectors[observed_inds, :]
        
        self.max_steps = max_steps

        self.best_reward = None
        self.best_action = None
        self.done = False
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        self.done = False
        self.best_reward = None
        self.best_action = None

        self.observed_inds = self.initial_inds
        self.observed_vectors = self.all_vectors[self.observed_inds, :]

        self.candidate_inds = self.all_actions[~jnp.isin(self.all_actions, self.initial_inds)]
        self.candidates = self.all_vectors[self.candidate_inds, :]
        self.candidate_targets = self.all_targets[self.candidate_inds]

        # self.candidate_data = deepcopy(self.initial_candidate_data)
        self.action_space = spaces.Discrete(len(self.candidate_inds))
        
        return self.initial_data, self.initial_targets

    def step(self, actions):
        if self.done:
            raise ValueError("Episode is done. Please call reset().")
        
        # Actions are indexes of the current candidate_inds list
        inds = self.candidate_inds[actions]
        
        # Take action
        observations = self.all_vectors[inds]
        
        # Evaluate the vectors using lookup data
        true_rewards = self.all_targets[inds]
        
        # Update best true_rewards and action
        if self.best_reward is None or jnp.max(true_rewards) > self.best_reward:
            self.best_reward = jnp.max(true_rewards)
            self.best_action = actions[jnp.argmax(true_rewards)]

        # Update episode termination conditions
        self.current_step += 1

        if self.current_step >= self.max_steps:
            self.done = True

        # Update the observed and candidate search spaces
        self.observed_inds = jnp.hstack((self.observed_inds, inds))
        self.observed_vectors = self.all_vectors[self.observed_inds]

        # Reduce the candidate pool
        self.candidate_inds = self.all_actions[~jnp.isin(self.all_actions, self.observed_inds)]
        self.candidates = self.all_vectors[self.candidate_inds]
        self.candidate_targets = self.all_targets[self.candidate_inds]

        self.action_space = spaces.Discrete(len(self.candidate_inds))

        self.total_reward = jnp.sum(self.all_targets[self.observed_inds])

        return observations, true_rewards, self.done, {"target_idx": self.current_step, "best_reward": self.best_reward, "best_action": self.best_action}

class RLAgent:
    def __init__(self, 
                 env, 
                 learning_rate=0.001, 
                 epsilon=0.5, 
                 epsilon_decay=0.998, 
                 epsilon_min=0.05, 
                 batch_size=16):

        self.env = env
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Define RL Network
        self.model = self.build_model()
        
        # Initialize replay buffer
        self.replay_buffer = deque(maxlen=10000)
        
    def build_model(self):
        model = MODNetModel([[['refractive_index']]],
                    {'refractive_index': 1},
                    num_neurons=[[256],[64],[64],[32]],
                    n_feat=N_FEATS
                    )
        
        model.fit(self.env.initial_data)

        return model
    

    def reset(self, key):
        self.model = MODNetModel([[['refractive_index']]],
                    {'refractive_index': 1},
                    num_neurons=[[256],[64],[64],[32]],
                    n_feat=N_FEATS
                    )
        
        self.model.fit(self.env.initial_data)
        self.replay_buffer = deque(maxlen=10000)

    
    def remember(self, actions, predictions, rewards):
        for i in range(len(actions)):
            self.replay_buffer.append((actions[i], predictions[i], rewards[i]))
        
        # TODO epsilon scaling?
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
            
    def act(self, key, env):
        # Predict the value of every candidate material
        candidate_data = env.all_data.from_indices(env.candidate_inds)
        model_values = self.model.predict(candidate_data).to_numpy().flatten()

        # Generate exploit probabilities
        exploit_probs = random.uniform(key, shape=(self.batch_size,))
        exploit_count = jnp.sum(exploit_probs > self.epsilon)

        all_actions = jnp.arange(env.action_space.n)

        # Take each of the inds (returns all inds when exploit_count=0)
        if exploit_count > 0:
            exploit_inds = jnp.argsort(model_values)[-exploit_count:]
        else:
            exploit_inds = jnp.array([], dtype=int)

        mask = ~jnp.isin(all_actions, exploit_inds)
        remaining_actions = all_actions[mask]

        # Select the rest of the batch randomly
        explore_inds = random.choice(key, 
                             a=remaining_actions, 
                             shape=(self.batch_size - exploit_count,),
                             replace=False)
        
        inds = jnp.concatenate((exploit_inds, explore_inds))
        predictions = jnp.concatenate((model_values[exploit_inds], model_values[explore_inds]))

        # Back propogate these losses through the model
        self.model.fit(env.all_data.from_indices(inds))

        return inds, predictions

class MPAgent:
    def __init__(self, 
                 n_feats = 974, 
                 epsilon=0.2,
                 batch_size=16,
                 learning_rate=1e-4):
        self.n_feats = n_feats
        self.model = DRLModel()
        self.model_params = self.model.init(random.PRNGKey(0), jnp.ones((1, n_feats)))['params']

        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.model_params)

        self.replay_buffer = ReplayBuffer(batch_size)

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def forward_pass(self, state_action_pairs):
        
        return DRLModel().apply({'params': self.model_params}, state_action_pairs)
    
    def reset(self, key):
        self.model = DRLModel()
        self.model_params = self.model.init(key, jnp.ones((1, self.n_feats)))['params']
        self.opt_state = self.optimizer.init(self.model_params)
        self.replay_buffer = ReplayBuffer(self.batch_size)

    def loss(self, state_action_pairs, targets):
        predictions = DRLModel().apply({'params': self.params}, state_action_pairs)
        return mse_loss(targets, predictions) 
    
    def act(self, key, state_action_pairs):
        # Generate all q values
        predicted_reward = self.forward_pass(state_action_pairs).flatten()
        
        # Generate exploit probabilities
        exploit_probs = random.uniform(key, shape=(self.batch_size,))
        exploit_count = jnp.sum(exploit_probs > self.epsilon)

        # Note env.action_space is < full inds, must add selection_offset for consistency
        available_actions = jnp.arange(state_action_pairs.shape[0])

        # Take each of the inds (returns all inds when exploit_count=0)
        if exploit_count > 0:
            exploit_inds = jnp.argsort(predicted_reward)[-exploit_count:]
        else:
            exploit_inds = jnp.array([], dtype=int)

        explore_inds = ~jnp.isin(available_actions, exploit_inds)
        remaining_actions = available_actions[explore_inds]

        # Select the rest of the batch randomly
        explore_inds = random.choice(key, 
                             a=remaining_actions, 
                             shape=(self.batch_size - exploit_count,),
                             replace=False)

        actions = jnp.concatenate((exploit_inds, explore_inds))

        predictions = jnp.concatenate((predicted_reward[exploit_inds], predicted_reward[explore_inds]))

        return actions, predictions


    def train_step(self, batch):
        # predictions = DRLModel().apply({'params': params}, state_action_pairs)
        # loss = mse_loss(targets, predictions)
        states = jnp.tile(batch[0][0], (self.batch_size, 1)) 
        state_action_pairs = jnp.hstack((states, jnp.array([x[3] for x in batch])))
        rewards = jnp.array([x[2] for x in batch])
        
        # Compute gradients with respect to the parameters
        forward_pass_fn = value_and_grad(lambda p: mse_loss(rewards, 
                                            self.model.apply({'params': p}, state_action_pairs)))
        loss, grads = forward_pass_fn(self.model_params)
        
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.model_params = optax.apply_updates(self.model_params, updates)

        state_action_pairs
        
        return self.model_params, self.opt_state, loss
    
class DQNAgent:
    def __init__(self, 
                 n_feats = 974, 
                 epsilon=0.2,
                 gamma = 0.99,
                 batch_size=16,
                 learning_rate=1e-4):
        
        self.n_feats = n_feats
        self.model = DRLModel()
        self.model_params = self.model.init(random.PRNGKey(0), jnp.ones((1, n_feats)))['params']

        self.target = DRLModel()
        self.target_params = self.target.init(random.PRNGKey(0), jnp.ones((1, n_feats)))['params']

        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.model_params)

        self.replay_buffer = ReplayBuffer(batch_size)

        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
    
    def reset(self, key):
        self.model = DRLModel()
        self.model_params = self.model.init(key, jnp.ones((1, self.n_feats)))['params']
        self.opt_state = self.optimizer.init(self.model_params)
        self.replay_buffer = ReplayBuffer(self.batch_size)
        
    def forward_pass(self, state_action_pairs):
        
        return DRLModel().apply({'params': self.model_params}, state_action_pairs)
    
    def loss(self, state_action_pairs, targets):
        predictions = DRLModel().apply({'params': self.params}, state_action_pairs)
        return mse_loss(targets, predictions) 
    
    def update_target_network(self):
        self.target_params = self.model_params
    
    def act(self, key, state_action_pairs):
        # Generate all q values
        predicted_reward = self.forward_pass(state_action_pairs).flatten()
        
        # Generate exploit probabilities
        exploit_probs = random.uniform(key, shape=(self.batch_size,))
        exploit_count = jnp.sum(exploit_probs > self.epsilon)

        # Note env.action_space is < full inds, must add selection_offset for consistency
        available_actions = jnp.arange(state_action_pairs.shape[0])

        # Take each of the inds (returns all inds when exploit_count=0)
        if exploit_count > 0:
            exploit_inds = jnp.argsort(predicted_reward)[-exploit_count:]
        else:
            exploit_inds = jnp.array([], dtype=int)

        explore_inds = ~jnp.isin(available_actions, exploit_inds)
        remaining_actions = available_actions[explore_inds]

        # Select the rest of the batch randomly
        explore_inds = random.choice(key, 
                             a=remaining_actions, 
                             shape=(self.batch_size - exploit_count,),
                             replace=False)

        actions = jnp.concatenate((exploit_inds, explore_inds))

        predictions = jnp.concatenate((predicted_reward[exploit_inds], predicted_reward[explore_inds]))

        return actions, predictions

    def train_step(self, batch):
        states = jnp.tile(batch[0][0], (len(batch), 1)) 
        state_action_pairs = jnp.hstack((states, jnp.array([x[3] for x in batch])))
        rewards = jnp.array([x[2] for x in batch])

        # Calculate target Q-values
        next_q_values = self.target.apply({'params': self.target_params}, state_action_pairs)
        max_next_q_values = jnp.max(next_q_values, axis=1)
        target_q_values = rewards + self.gamma * max_next_q_values

        # Calculate the current Q-values for the model 
        action_q_values = self.model.apply({'params': self.model_params}, state_action_pairs)

        # Compute gradients with respect to the parameters
        loss, grads = value_and_grad(lambda p: mse_loss(target_q_values, 
                                                            action_q_values))(self.model_params)
        
        # Update model parameters using gradients
        updates, opt_state = self.optimizer.update(grads, self.opt_state, self.model_params)
        self.model_params = optax.apply_updates(self.model_params, updates)
        self.opt_state = opt_state
        
        return self.model_params, self.opt_state, loss
    

data = MODData.load('mod.data_refeatselec_v4')
n_starting = 32
n = len(data.get_featurized_df())

####################################################
# Training loop
####################################################

ensembles = 1
n_runs = 1
learning_rate = 1e-3
batch_size = 4
target_copy_interval = 3
n_episodes = 5

key = random.PRNGKey(0)
key, subkey = random.split(key)

for agent in range(3):
    if agent != 2:
        continue
    try:
        res = pk.load(open(f'results.pk', 'rb'))
    except:
        res = {}
    
    if agent not in res: res[agent] = {}

    for epsilon in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        if epsilon not in res[agent]: res[agent][epsilon] = {}

        for run in range(n_runs):            
            if run not in res[agent][epsilon] : res[agent][epsilon][run] = {} 

            starting_inds = random.choice(key=subkey, 
                                a=n, 
                                shape=(n_starting,))

            # Used for MODNet agent
            initial_data = data.from_indices(starting_inds)

            # Used by vector agents
            X = data.get_featurized_df()[data.optimal_features].values
            non_nan_inds = jnp.where(~jnp.isnan(X).any(axis=0))[0]

            X = X[:, non_nan_inds]
            y_true = data.targets.flatten()

            n, m = X.shape # n is number of samples, m is number of features

            if agent == 0:
                env = ModNetSearchEnv(data, starting_inds, max_steps=n_episodes)
                rl_agent = RLAgent(env, 
                                epsilon=epsilon,
                                batch_size=batch_size)
            
            elif agent == 1:
                env = VectorSearchEnv(X, y_true, starting_inds, max_steps=n_episodes)
                rl_agent = MPAgent(n_feats=2*m,
                                epsilon=epsilon,
                                batch_size=batch_size,
                                learning_rate=learning_rate)
                
            elif agent == 2:
                env = VectorSearchEnv(X, y_true, starting_inds, max_steps=n_episodes)
                rl_agent = DQNAgent(n_feats=2*m,
                                epsilon=epsilon,
                                batch_size=batch_size,
                                learning_rate=learning_rate)
            
            for ensemble in range(ensembles):
                if agent > 0:
                    x_init, y_init = env.reset()
                    state_init = jnp.average(x_init, axis=0)
                    states = jnp.tile(state_init, (x_init.shape[0], 1)) # Repeat this state for each action
                    state_action_pairs = jnp.hstack((states, x_init))

                    batch = [(state_init, action, y_init[i], x_init[i], 0) for i, action in enumerate(x_init)]

                    print("Performing Warmup")

                    for i in tqdm(range(50)):
                        updated_params, updated_opt_state, loss = rl_agent.train_step(batch) 
                else:
                    states = env.reset()

                rl_agent.reset(subkey)

                done = False
                total_reward = 0
                best = 0
                i = 0

                while not env.done:
                    key, subkey = random.split(subkey)

                    if agent == 0:
                        # Choose actions
                        actions, predictions = rl_agent.act(subkey, env)
                        
                        # Update environment and get rewards
                        next_states, rewards, done, info = env.step(actions)
                        
                        # Store experience in replay buffer
                        rl_agent.remember(actions, predictions, rewards)
                        
                        # Get the witnessed rewards
                        all_rewards = jnp.array([x[2] for x in rl_agent.replay_buffer])

                        # Update total reward
                        total_reward = jnp.sum(all_rewards) 

                        if jnp.max(all_rewards) > best:
                            best = jnp.max(rewards)
                        
                        print(f'Run: {run}, eps: {epsilon}, Ensemble: {ensemble}, Iteration: {i}, Highest Prediction: {jnp.max(predictions):.3f}, Actual Best {jnp.max(rewards):.3f}, Total Reward: {total_reward:.3f}, Materials Discovered: {len(rl_agent.replay_buffer)}, Best: {best:.3f}')

                    else:
                        candidates = env.candidates
                        observed = env.observed_vectors
                        
                        # Calculate mean pooled state, and tile for each possible action
                        pooled_state = jnp.average(observed, axis=0)
                        states = jnp.tile(pooled_state, (candidates.shape[0], 1)) 
                        state_action_pairs = jnp.hstack((states, candidates))
                        
                        # Find all the actions and predicted rewards 
                        actions, predicted_reward = rl_agent.act(subkey, state_action_pairs)
                        next_candidates, rewards, done, info = env.step(actions)

                        batch = [(pooled_state, action, rewards[i], next_candidates[i], done) for i, action in enumerate(actions)]
                        rl_agent.replay_buffer.add(batch)

                        # Now that we have our chosen states, back propogate the loss through the model
                        rl_agent.train_step(batch)

                        # Copy weights and perform replay for the Q network
                        if env.current_step % target_copy_interval == 0 and agent == 2:
                            # Replay past experiences
                            batch = rl_agent.replay_buffer.sample(subkey)
                            rl_agent.train_step(batch)
                            rl_agent.update_target_network()

                        print(f'Run: {run}, eps: {epsilon}, Ensemble: {ensemble}, Iteration: {env.current_step}, Highest Reward: {env.best_reward:.3f}, Total Reward: {env.total_reward:.3f}, Materials Discovered: {len(rl_agent.replay_buffer)}, Average: {env.total_reward/len(rl_agent.replay_buffer):.3f}')

                    i += 1
                
                # Print ensemble information
                print(f"Ensemble: {ensemble}, Total Reward: {env.total_reward}, Best: {env.best_reward}, Average: {env.total_reward/len(rl_agent.replay_buffer)}")

                # Save results
                res[agent][epsilon][run][ensemble] = rl_agent.replay_buffer

                pk.dump(res, open(f'results.pk', 'wb'))