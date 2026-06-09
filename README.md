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

```python

```

---

## Output

```text

```

---

## Output Graph

The histogram displays the estimated state-value function for all states in the FrozenLake environment after TD learning.

---

## Result

Thus, the TD Prediction algorithm was successfully implemented in the FrozenLake environment to estimate the state-value function. The agent learned the value of each state through continuous interaction with the environment using Temporal Difference learning.
