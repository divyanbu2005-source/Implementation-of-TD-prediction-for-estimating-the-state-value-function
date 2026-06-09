# Ex-7: TD Prediction for Estimating the State-Value Function using FrozenLake Environment

## Aim:

To implement the Temporal Difference (TD) Prediction algorithm for estimating the state-value function in the FrozenLake environment using Reinforcement Learning.

---

## Algorithm:

### TD Prediction Algorithm:

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

## Program:

```python

# ============================================================
# TD PREDICTION FOR ESTIMATING STATE-VALUE FUNCTION
# ============================================================

import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Create Environment
env = gym.make("FrozenLake-v1", is_slippery=False)

# Parameters
alpha = 0.1
gamma = 0.9
episodes = 5000

# State Value Function
V = defaultdict(float)

# Random Policy
def policy(state):
    return env.action_space.sample()

# TD Prediction Algorithm
for ep in range(episodes):

    state, _ = env.reset()

    done = False

    while not done:

        action = policy(state)

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        # TD Target
        td_target = reward + gamma * V[next_state]

        # TD Error
        td_error = td_target - V[state]

        # Update State Value
        V[state] = V[state] + alpha * td_error

        state = next_state

# Print State Values
print("\nTD State Value Function:\n")

for s in range(env.observation_space.n):
    print(f"State {s}: {V[s]:.4f}")

# ------------------------------------------------
# Histogram Plot
# ------------------------------------------------

states = list(range(env.observation_space.n))
values = [V[s] for s in states]

plt.figure(figsize=(10,5))

plt.bar(states, values)

plt.xlabel("States")
plt.ylabel("Estimated State Value")
plt.title("TD Prediction State Value Function")

plt.show()

```

---

## Output:

```text

TD State Value Function:

State 0: 0.0060
State 1: 0.0065
State 2: 0.0143
State 3: 0.0060
State 4: 0.0087
State 5: 0.0000
State 6: 0.0399
State 7: 0.0000
State 8: 0.0156
State 9: 0.0531
State 10: 0.1214
State 11: 0.0000
State 12: 0.0000
State 13: 0.1270
State 14: 0.3672
State 15: 0.0000

```

---

## Output Graph:

The histogram displays the estimated state-value function for all states in the FrozenLake environment after TD learning.

<img width="855" height="468" alt="image" src="https://github.com/user-attachments/assets/45aed50e-7841-4db9-aef3-93da94d92c52" />


---

## Result:

Thus, the TD Prediction algorithm was successfully implemented in the FrozenLake environment to estimate the state-value function. The agent learned the value of each state through continuous interaction with the environment using Temporal Difference learning.
