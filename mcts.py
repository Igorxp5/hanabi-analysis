import math

from typing import Dict, List

import numpy as np

from agent import HanabiAgent

from hanabi.game import Hanabi
from hanabi.models import Player

STATE_EMPTY_CARD_SLOT = -1
STATE_EMPTY_CLUE_SLOT = -2


class HanabiMcts:
    def __init__(self, agent: HanabiAgent, degree_exploration: float):
        self._agent = agent
        self.degree_exploration = degree_exploration

        self._state_visits: Dict[bytes, int] = dict()
        self._state_action_visits: Dict[bytes, np.ndarray] = dict()
        self._policies: Dict[bytes, np.ndarray] = dict()
        self._state_action_values: Dict[bytes, np.ndarray] = dict()

    def simulate(self, hanabi_game: Hanabi):
        state = HanabiAgent.get_game_state(hanabi_game)
        state_key = state.tobytes()

        if not hanabi_game.is_alive():
            return self.get_state_value(hanabi_game)

        if state_key not in self._state_visits:
            # First visit
            self._state_visits[state_key] = 0
            valid_actions = HanabiAgent.get_valid_actions(hanabi_game)
            self._state_action_visits[state_key] = np.zeros(valid_actions.shape)
            self._policies[state_key] = self._agent.get_policy(hanabi_game) * valid_actions

            if not np.sum(self._policies[state_key]):
                self._policies[state_key] = valid_actions

            self._state_action_values[state_key] = np.zeros(valid_actions.shape)
            return self.get_state_value(hanabi_game)
        else:
            policy = self._select_action_by_ucb(hanabi_game)
            hanabi_copy = hanabi_game.copy()
            HanabiAgent.get_policy_action_function(hanabi_copy, policy)()
            state_action_value = self.simulate(hanabi_copy)

            action_index = np.argmax(policy)
            self._state_visits[state_key] += 1
            self._state_action_visits[state_key][action_index] += 1
            action_visits = self._state_action_visits[state_key][action_index]
            self._state_action_values[state_key][action_index] = (action_visits * self._state_action_values[state_key][action_index] + state_action_value) / (action_visits + 1)

            return state_action_value

    def get_policy(self, hanabi_game: Hanabi):
        # Policy is based on amount of visits. As MCTS prefers to vist the best actions.
        policy = self._agent.get_valid_actions(hanabi_game)
        state_key = HanabiAgent.get_game_state(hanabi_game).tobytes()
        for action_index in range(policy.size):
            if policy[action_index]:
                policy[action_index] = self._state_action_visits[state_key][action_index]
        return policy / (np.sum(policy) or 1)
    
    def get_state_value(self, hanabi_game: Hanabi):
        if hanabi_game.is_alive():
            return (hanabi_game.points / hanabi_game.max_points) - (hanabi_game.bombs / hanabi_game._max_bombs)
        return self._agent.get_estimated_reward(hanabi_game)

    def _select_action_by_ucb(self, hanabi_game: Hanabi) -> np.ndarray:
        valid_actions = self._agent.get_valid_actions(hanabi_game)
        state = HanabiAgent.get_game_state(hanabi_game)
        state_key = state.tobytes()
        max_ucb_action = max(np.nonzero(valid_actions)[0], key=lambda i: self._get_action_ucb(state_key, i))
        policy = np.zeros(valid_actions.shape)
        policy[max_ucb_action] = 1
        return policy

    def _get_action_ucb(self, state_key: bytes, action_index: int):
        bound = math.sqrt(self._state_visits[state_key]) / (1 + self._state_action_visits[state_key][action_index])
        action_probability = self._get_action_probability(state_key, action_index)
        return self._state_action_values[state_key][action_index] + self.degree_exploration * action_probability * bound

    def _get_action_probability(self, state_key: bytes, action_index: int):
        return self._policies[state_key][action_index] / np.sum(self._policies[state_key])
    