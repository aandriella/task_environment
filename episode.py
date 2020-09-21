"""
Episodes representing expert demonstrations and automated generation
thereof.
"""


import numpy as np
from itertools import chain
import itertools
import os
import time
import sys

class Episode:
    """
    A episode consisting of states, corresponding actions, and outcomes.

    Args:
        transitions: The transitions of this episode as an array of
            tuples `(state_from, action, state_to)`. Note that `state_to` of
            an entry should always be equal to `state_from` of the next
            entry.
    """

    def __init__(self, states=[]):
        self._t = list()
        for s in states:
            self._t.append(tuple(s))

    def transition(self, state_from, action, state_to):
        self._t.append((state_from, action, state_to))

    def transitions(self):
        """
        The transitions of this episode.

        Returns:
            All transitions in this episode as array of tuples
            `(state_from, action, state_to)`.
        """
        return self._t

    def states(self):
        """
        The states visited in this episode.

        Returns:
            All states visited in this episode as iterator in the order
            they are visited. If a state is being visited multiple times,
            the iterator will return the state multiple times according to
            when it is visited.
        """
        return map(lambda x: x[0], chain(self._t, [(self._t[-1][2], 0, 0)]))

    def __repr__(self):
        return "EpisodeGenerator({})".format(repr(self._t))

    def __str__(self):
        return "{}".format(self._t)





    def get_states(self, states, initial_state):
        states_list = list(itertools.product(*states))
        states_list.insert(0, initial_state)
        return states_list


    def state_from_point_to_index(self, states, point):
      return states.index(tuple(point))



    def state_from_index_to_point(self, state_tuple, index):
        return state_tuple[index]

    def load_episodes(self, file):
        '''
        It returns the episodes related to the saved file
        :param file:
        :param episode: look at main.py
        :param sol_per_pop: look at main.py
        :return: a list of episodes
        '''
        print("LOADING...")

        trajs = list()
        with open(file, "rb") as f:
            traj = np.load(f, allow_pickle=True)
            for t in range(len(traj)):
                trajs.append(Episode(traj[t]))
                print("loaded traj ", t)
            f.close()
        for t in trajs:
            print(t._t)
        return trajs


    def generate_statistics(self, state_list, action_space, episodes):
        '''
        This function computes the state x state x action matrix that
        corresponds to the transition table we will use later
        '''
        print(state_list)
        n_states = len(state_list)
        n_actions = len(action_space)

        #create a matrix state x state x action
        table = np.zeros(shape=(n_states, n_states,  n_actions))
        start_time = time.time()
        s1, s2, a = range(n_states), range(n_states), range(n_actions)
        for s_from in s1:
            for act in a:
                for s_to in s2:
                    #convert to coord
                    s_from_coord = self.state_from_index_to_point(state_list, s_from)
                    s_to_coord = self.state_from_index_to_point(state_list, s_to)
                    #print("from:", s_from_coord," to:", s_to_coord)
                    #print()
                    for traj in episodes:
                        if (s_from, act, s_to) in traj._t:
                            table[s_from, s_to, act] += 1
        elapsed_time = time.time()-start_time
        print("processing time:{}".format(elapsed_time))
        return table


    def compute_probabilities(self, transition_matrix, terminal_state, state_space):
        """
        We compute the transitions for each state_from -> action -> state_to
        :param transition_matrix:  matrix that has shape n_states x n_states x action
        :return:
        """
        n_state_from, n_state_to, n_actions = transition_matrix.shape
        transition_matrix_with_prob = np.zeros((n_state_from, n_state_to, n_actions))

        for s_from in range(n_state_from):
            s_in_prob = list()
            sum_over_prob = 0
            #get the episode from s_from to all the possible state_to given the 5 actions
            #get all the occurrence on each column and compute the probabilities
            #remember for each column the sum of probabilities has to be 1
            for a in range(n_actions):
                trans_state_from = list(zip(*transition_matrix[s_from]))[a]
                #needs to be done to avoid nan (0/0)

                sum_over_prob = sum(trans_state_from) if sum(trans_state_from)>0 else sys.float_info.min

                s_in_prob.append(list(map(lambda x: x/sum_over_prob, trans_state_from)))

            transition_matrix_with_prob[s_from][:][:] = np.asarray(s_in_prob).T

        for state in terminal_state:
            state_idx = self.state_from_point_to_index(state_space, state)
            transition_matrix_with_prob[state_idx][state_idx][0] = 1

        return transition_matrix_with_prob


    def read_transition_matrix(self, file):
        print("Loading trans matrix...")
        fileinfo = os.stat(file)
        trans_matrix = list()
        with open(file, "rb") as f:
            trans_matrix = np.load(f, allow_pickle=True)

        #trans_matrix_reshaped = np.asarray(trans).reshape(n_states, n_states, n_actions)
        print("Done")
        return trans_matrix


def main():

    file_path = "/home/aandriella/Documents/Codes/MY_FRAMEWORK/BN_GenerativeModel/results/1/episodes.npy"
    ep = Episode()
    episodes = ep.load_episodes(file_path)
    initial_state = (1, 1, 0)
    n_max_attempt = 5
    task_length = 6
    # Environment setup for RL agent assistance
    action_space = ['LEV_0', 'LEV_1', 'LEV_2', 'LEV_3', 'LEV_4', 'LEV_5']
    user_actions_state = [-1, 0, 1]
    final_states = [(task_length, a, u)   for a in range(1, n_max_attempt) for u in range(-1, 2) ]
    # defintion of state space
    attempt = [i for i in range(1, n_max_attempt)]
    game_state = [i for i in range(1, task_length+1)]
    user_actions = [i for i in (user_actions_state)]
    states_space = (game_state, attempt, user_actions)  # , task_levels)

    env = Environment(action_space, initial_state, final_states, user_actions, states_space,
                                 task_length, n_max_attempt, timeout=0, n_levels_assistance=6)
    #
    trans_matrix = ep.generate_statistics(env.states, env.action_space, episodes)
    path_trans_matrix_occ = "/home/aandriella/Documents/Codes/MY_FRAMEWORK/BN_GenerativeModel/results/1/trans_matrix_occ.npy"
    path_trans_matrix_prob = "/home/aandriella/Documents/Codes/MY_FRAMEWORK/BN_GenerativeModel/results/1/trans_matrix_prob.npy"
    terminal_states = [env.point_to_index(state) for state in final_states]


    # save the episode on a file
    with open(path_trans_matrix_occ, "ab") as f:
        np.save(f, trans_matrix)
        f.close()
    trans_matrix_occ = ep.read_trans_matrix(path_trans_matrix_occ)
    print(trans_matrix_occ.shape)
    trans_matrix_prob = ep.compute_probabilities(trans_matrix_occ, terminal_states)
    # save the episode on a file
    with open(path_trans_matrix_prob, "ab") as f:
        np.save(f, trans_matrix_prob)
        f.close()

    #prob = read_trans_matrix(path_trans_matrix_prob, 0, 0)



if __name__ == "__main__":
    main()
