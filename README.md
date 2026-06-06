# TD Prediction for Estimating the State-Value Function using FrozenLake Environment

## Aim

To implement the Temporal Difference (TD) Prediction algorithm for estimating the state-value function in the FrozenLake environment using Reinforcement Learning.

---

## Algorithm

### TD Prediction Algorithm

1. Import the required libraries.
2. Create the FrozenLake environment using OpenAI Gym.
3. Initialize:
   - Learning rate \( \alpha \)
   - Discount factor \( \gamma \)
   - Number of episodes
   - State-value function \( V(s) \)
4. Define a random policy for action selection.
5. For each episode:
   - Reset the environment.
   - Repeat until the episode ends:
     - Select an action using the policy.
     - Perform the action and observe:
       - Next state
       - Reward
       - Terminal condition
     - Compute TD Target:

\[
TD\ Target = R + \gamma V(S')
\]

     - Compute TD Error:

\[
TD\ Error = TD\ Target - V(S)
\]

     - Update the state-value function:

\[
V(S) = V(S) + \alpha \times TD\ Error
\]

     - Move to the next state.
6. Print the estimated state values.
7. Plot the state-value function using a histogram.

---

## Program

```
# ============================================================
# TD PREDICTION FOR ESTIMATING STATE-VALUE FUNCTION
# ============================================================

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Create Environment
env = gym.make("FrozenLake-v1", is_slippery=False)

# Parameters
alpha = 0.1
gamma = 0.9
episodes = 5000

# State Value Function
n_states = env.observation_space.n
V = np.zeros(n_states)

# Random Policy
def policy(state):
    return env.action_space.sample()

# TD(0) Prediction
for episode in range(episodes):

    state, _ = env.reset()
    done = False

    while not done:

        action = policy(state)

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        # TD Target
        if done:
            td_target = reward
        else:
            td_target = reward + gamma * V[next_state]

        # TD Error
        td_error = td_target - V[state]

        # Update State Value
        V[state] += alpha * td_error

        state = next_state

# ============================================================
# Print State Values
# ============================================================

print("\nTD State Value Function:\n")

for s in range(n_states):
    print(f"State {s}: {V[s]:.4f}")

# ============================================================
# Histogram Plot
# ============================================================

plt.figure(figsize=(10,5))
plt.bar(range(n_states), V)

plt.xlabel("States")
plt.ylabel("Estimated State Value")
plt.title("TD(0) Prediction - FrozenLake State Values")
plt.xticks(range(n_states))

plt.show()
```
## Output
<img width="1020" height="433" alt="{67D03289-D429-4043-8E82-54475E5B2F98}" src="https://github.com/user-attachments/assets/e269bc91-5e88-4307-b363-5ba782a3d94b" />

## Output Graph

The histogram displays the estimated state-value function for all states in the FrozenLake environment after TD learning.

<img width="1132" height="582" alt="{894FD221-CB50-455E-A9E0-EFC0454CD660}" src="https://github.com/user-attachments/assets/78b33cb3-7c93-4c12-bcdb-a007976722fe" />

## Result

Thus, the TD Prediction algorithm was successfully implemented in the FrozenLake environment to estimate the state-value function. The agent learned the value of each state through continuous interaction with the environment using Temporal Difference learning.
