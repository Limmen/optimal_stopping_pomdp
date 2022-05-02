from typing import Tuple
import numpy as np


def one_step_lookahead(state, V, num_actions, num_states, T, discount_factor, R, V_l_1: np.ndarray, l: int) \
        -> np.ndarray:
    """
    Performs a one-step lookahead for value iteration
    :param state: the current state
    :param V: the current value function
    :param num_actions: the number of actions
    :param num_states: the number of states
    :param T: the transition kernel
    :param discount_factor: the discount factor
    :param R: the table with rewards
    :param next_state_lookahead: the next state lookahead table
    :param V_l_1: the value function for the future when taking a stop action
    :return: an array with lookahead values
    """
    A = np.zeros(num_actions)
    for a in range(num_actions):
        reward = R[a][state]
        if l > 1 and a == 1:
            reward += V_l_1[state]
        for next_state in range(num_states):
            prob = T[a][state][next_state]
            A[a] += prob * (reward + discount_factor * V[next_state])
    return A


def value_iteration(T: np.ndarray, num_states: int, num_actions: int, R: np.ndarray, V_l_1: np.ndarray, l: int,
                    theta=0.0001, discount_factor=1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    An implementation of the Value Iteration algorithm
    :param T: the transition kernel T
    :param num_states: the number of states
    :param num_actions: the number of actions
    :param state_to_id: the state-to-id lookup table
    :param HP: the table with hack probabilities
    :param R: the table with rewards
    :param next_state_lookahead: the next-state-lookahead table
    :param theta: convergence threshold
    :param discount_factor: the discount factor
    :param V_l_1: the value function for the future when taking a stop action
    :return: (greedy policy, value function)
    """
    V = np.zeros(num_states)

    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(num_states):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V, num_actions, num_states, T, discount_factor, R, V_l_1, l)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function. Ref: Sutton book eq. 4.10.
            V[s] = best_action_value

        # print("delta:{}".format(delta))
        # Check if we can stop
        if delta < theta:
            break

    # Create a deterministic policy using the optimal value function
    policy = np.zeros([num_states, num_actions * 2])
    for s in range(num_states):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V, num_actions, num_states, T, discount_factor, R, V_l_1=V_l_1, l=l)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s, best_action] = 1.0
        # policy[s][2] = A[0]
        # policy[s][3] = A[1]

    return V, policy
