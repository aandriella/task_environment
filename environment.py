import itertools
import numpy as np

from episode import Episode

"""
This class contains the definition of the current assistive domain
"""


class Environment:
  def __init__(self, action_space, initial_state, goal_states, user_actions, states, rewards,
               task_complexity=3, task_length=10, n_max_attempt_per_object=5,
               timeout=20, n_levels_assistance=5):

    self.action_space = action_space
    self.initial_state = initial_state
    self.goal_states = goal_states
    self.user_actions = user_actions
    self.rewards = rewards
    self.states = self.get_states(states)
    self.task_complexity = task_complexity
    self.task_length = task_length
    self.task_progress = 1
    self.n_max_attempt_per_object = n_max_attempt_per_object
    self.n_attempt_per_object = 1
    self.n_total_attempt = 1
    self.timeout = timeout
    self.n_levels_assistance = n_levels_assistance

  def reset(self):
    self.task_progress = 1
    self.n_attempt_per_object = 0
    self.n_total_attempt = 0

  def get_actions_space(self):
    return self.action_space

  def get_states_list(self):
    return self.states

  def get_states(self, states):
    states_list = list(itertools.product(*states))
    # states_list.insert(0, self.initial_state)
    return states_list

  def point_to_index(self, point):
    return self.states.index(tuple(point))

  def get_initial_state(self):
    return self.initial_state

  def is_goal_state(self, state):
    return (state) in self.goal_states

  def get_task_length(self):
    return self.task_length

  def get_task_level(self):
    return self.task_level

  def get_task_progress(self):
    return self.task_progress

  def get_n_objects(self):
    return self.n_objects

  def get_n_attempt_per_object(self):
    return self.n_attempt_per_object

  def get_n_total_attempt(self):
    return self.n_total_attempt

  def get_n_max_attempt_per_object(self):
    return self.n_max_attempt_per_object

  def get_n_levels_assistance(self):
    return self.n_levels_assistance

  def get_timeout(self):
    return self.timeout

  def set_task_length(self, other):
    self.task_length = other

  def set_task_level(self, other):
    self.task_level = other

  def set_task_progress(self, other):
    self.task_progress = other

  def set_n_objects(self, other):
    self.n_objects = other

  def set_n_attempt_per_object(self, other):
    self.n_attempt_per_object = other

  def set_n_total_attempt(self, other):
    self.n_total_attempt = other

  def set_n_max_attempt_per_object(self, other):
    self.n_max_attempt_per_object = other

  def set_timeout(self, other):
    self.timeout = other

  def gen_transition_matrix(self, episodes, state_space_list, action_space_list, final_state_list,
                            path_trans_matrix_occ,
                            path_trans_matrix_prob):
    '''
    This function generate the transition matrix function
    Args:
      episodes: the list of episodes
      state_space_list: the list of states
      action_space_list: the list of actions
      final_state_list: the list of final states
      path_trans_matrix_occ: the file with the occ matrix
      path_trans_matrix_prob: the file with the matrix prob
    Return:
      the transition matrix probabilities
    '''
    trans_matrix = Episode.generate_statistics(episodes, state_space_list, action_space_list)
    # save the episode on a file
    with open(path_trans_matrix_occ, "ab") as f:
      np.save(f, trans_matrix)
      f.close()
    trans_matrix_occ = Episode.read(path_trans_matrix_occ)
    trans_matrix_prob = Episode.compute_probabilities(trans_matrix_occ, final_state_list)
    # save the episode on a file
    with open(path_trans_matrix_prob, "ab") as f:
      np.save(f, trans_matrix_prob)
      f.close()
    return trans_matrix_prob


  def step(self, current_state, outcome, assistance_level, reward_matrix):
    '''This function compute the next state and the current reward
      Args:
      outcome: is the outcome of the user's action
      assistance_level: is the level of assistance given by the robot
      reward_matrix: if there is a file then use it for as reward
      Returns:
      :next_state
      :reward
      :done if reached the final state
    '''

    reward = 0
    done = False
    user_action = 0

    if self.task_progress >= 0 and self.task_progress <= self.task_length / 3:
      self.task_complexity = 1
    elif self.task_progress > self.task_length / 3 and self.task_progress <= 2 * self.task_length / 3:
      self.task_complexity = 2
    elif self.task_progress > 2 * self.task_length / 3 and self.task_progress < self.task_length:
      self.task_complexity = 3
    elif self.task_progress == self.task_length:
      self.task_complexity = 4


    if outcome[0] == "max_attempt":
      user_action = 1
      reward = -1
      self.n_attempt_per_object = 1
      self.n_total_attempt += 1
      self.task_progress += 1

    elif outcome[0] == "wrong_action":
      user_action = -1
      reward = 0  # * (self.task_progress) * (self.n_attempt_per_object+1) * (assistance_level+1)
      self.n_attempt_per_object += 1
      self.n_total_attempt += 1

    elif outcome[0] == "timeout":
      user_action = 0
      reward = -1
      self.n_attempt_per_object += 1
      self.n_total_attempt += 1
    elif outcome[0] == "correct_action":
      user_action = 1
      reward = 0.3 * (self.n_levels_assistance - assistance_level - 1) / (self.n_levels_assistance) + 0.3 * (
          self.n_max_attempt_per_object - self.n_attempt_per_object + 1) / (self.n_max_attempt_per_object) + 0.3 * (
                   self.task_length - self.task_progress) / (self.task_length) + 0.1 * (
                   30 - outcome[1]) / 30
      self.n_attempt_per_object = 1
      self.n_total_attempt += 1
      self.task_progress += 1
    else:
      assert Exception("The outcome index is not right, please check out the documentation")

    next_state = (self.task_complexity, self.n_attempt_per_object, user_action)  # , self.task_level)

    if self.is_goal_state(next_state):
      done = True

    if reward_matrix != "":
      reward = reward_matrix(current_state, assistance_level, next_state)

    return next_state, reward, done


def __str__(self):
  return "RL Method" + self.rl_method + "action_space[" + str(self.action_space)[1:-1] + "] feedback [" + str(
    self.feedback)[1:-1] + "] reward [" + str(self.reward)[1:-1] + "] state [" + str(self.state)[
                                                                                 1:-1] + "] max_attempt:" + str(
    self.max_attempt) + " game complexity: " + str(self.game_complexity)

