from typing import Callable, Tuple, Iterable
from functools import partial

import numpy as np

import tensorflow as tf

from hanabi.game import Hanabi
from hanabi.models import Player, Card, Clue

STATE_EMPTY_CARD_SLOT = -1
STATE_EMPTY_CLUE_SLOT = -2

TrainingExample = Tuple[np.ndarray, np.ndarray, float]


class HanabiAgent:
    def __init__(self, cards_set: Tuple[Card], clues_set: Tuple[Clue],
                total_players: 2, initial_hand_size: int = 5,
                learning_rate=0.001, batch_size=64, epochs=10):
        players = tuple(Player(f'{i}') for i in range(total_players))
        hanabi_game = Hanabi(cards_set, clues_set, players, initial_hand_size)

        self._cards_set = cards_set
        self._clues_set = clues_set
        self._initial_hand_size = initial_hand_size
        self._colors = hanabi_game._colors
        self._unique_cards_by_color = hanabi_game._unique_cards_by_color
        self._total_players = total_players
        
        self._state_shape = HanabiAgent.get_game_state(hanabi_game, players[0]).shape
        self._actions_shape = HanabiAgent.get_actions_shape(hanabi_game)

        model_input = tf.keras.layers.Input(shape=self._state_shape)

        hidden_layer_1 = tf.keras.layers.Dense(units=1024, activation='relu')(model_input)
        hidden_layer_2 = tf.keras.layers.Dense(units=512, activation='relu')(hidden_layer_1)

        self._policy = tf.keras.layers.Dense(self._actions_shape[0], activation='softmax', name='policy')(hidden_layer_2)
        self._state_value = tf.keras.layers.Dense(1, activation='tanh', name='state_value')(hidden_layer_2)

        self._model = tf.keras.models.Model(inputs=model_input, outputs=[self._policy, self._state_value])
        self._model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=tf.keras.optimizers.Adam(learning_rate))

        self.batch_size = batch_size
        self.epochs = epochs
    
    def train(self, examples: Iterable[TrainingExample], verbose=None):
        states, policies, rewards = list(zip(*examples))
        states = np.asarray(states)
        policies = np.asarray(policies)
        rewards = np.asarray(rewards)

        self._model.fit(x=states, y=[policies, rewards], batch_size=self.batch_size,
                        epochs=self.epochs, verbose=verbose)
    
    def act(self, hanabi_game: Hanabi):
        policy = self.get_policy(hanabi_game)
        return HanabiAgent.get_policy_action_function(hanabi_game, policy)()

    def get_policy(self, hanabi_game: Hanabi) -> np.ndarray:
        input_state = HanabiAgent.get_game_state(hanabi_game).reshape((1,) + self._state_shape)
        policy, _ = self._model.predict(input_state, verbose=None)
        return policy[0]

    def get_estimated_reward(self, hanabi_game: Hanabi) -> float:
        input_state = HanabiAgent.get_game_state(hanabi_game).reshape((1,) + self._state_shape)
        _, state_value = self._model.predict(input_state, verbose=None)
        return state_value[0]

    def export_model(self, filepath):
        return self._model.save_weights(filepath, save_format='h5')

    def load_model(self, filepath):
        self._model.load_weights(filepath)

    @staticmethod
    def get_policy_action_function(hanabi_game: Hanabi, policy: np.ndarray) -> Callable:
        play_offset = hanabi_game._initial_hand_size
        clue_color_offset = play_offset + hanabi_game._initial_hand_size
        clue_number_offset = clue_color_offset + (len(hanabi_game._colors) * (len(hanabi_game._players) - 1))

        action_index = np.argmax(policy)

        if action_index < play_offset:  # Discard a card
            return partial(hanabi_game.discard_card, action_index)
        elif action_index < clue_color_offset:  # Play a card
            return partial(hanabi_game.play_card, action_index - play_offset)
        elif action_index < clue_number_offset:  # Give a number clue to some player
            player_index = (action_index - clue_color_offset) // len(hanabi_game._colors)
            clue_color = hanabi_game._colors[(action_index - clue_color_offset) % len(hanabi_game._colors)]
            clue = Clue(color=clue_color)
            return partial(hanabi_game.give_clue, hanabi_game.other_players[player_index], clue)
        else:  # Give a color clue to some player
            player_index = (action_index - clue_number_offset) // hanabi_game._unique_cards_by_color
            color_number = ((action_index - clue_number_offset) % len(hanabi_game._colors)) + 1
            clue = Clue(number=color_number)
            return partial(hanabi_game.give_clue, hanabi_game.other_players[player_index], clue)

    @staticmethod
    def get_valid_actions(hanabi_game: Hanabi) -> np.ndarray:
        valid_actions = np.ones(HanabiAgent.get_actions_shape(hanabi_game))
        if hanabi_game.clues == 0:
            clue_offset = hanabi_game._initial_hand_size * 2  # discard/play card actions
            valid_actions[clue_offset:] = 0
        return valid_actions

    @staticmethod
    def get_actions_shape(hanabi_game: Hanabi) -> Tuple[int]:
        action_size = hanabi_game._initial_hand_size  # discard a card
        action_size += hanabi_game._initial_hand_size # play a card
        action_size += (len(hanabi_game._colors) + hanabi_game._unique_cards_by_color) * (len(hanabi_game._players) - 1)  # give a clue to a player
        return action_size,

    @staticmethod
    def get_game_state(hanabi_game: Hanabi, player_perspective: Player = None) -> np.ndarray:
        if player_perspective:
            player_perspective = hanabi_game.current_player

        cards_states_length = len(hanabi_game._cards_set)
        other_players_cards_length = (len(hanabi_game._players) - 1) * hanabi_game._initial_hand_size * 2
        player_clues_length = len(hanabi_game._players) * hanabi_game._initial_hand_size * 2
        available_clues_length = 1
        bombs_length = 1

        state_length = cards_states_length + other_players_cards_length + player_clues_length \
            + available_clues_length + bombs_length

        state = np.zeros(state_length)
        index = 0
        
        for card in hanabi_game._cards_set:
            state[index] = hanabi_game._cards_state[card].value
            index += 1

        for player in hanabi_game.other_players:
            player_cards = hanabi_game.get_player_cards(player)
            for j in range(hanabi_game._initial_hand_size):
                card_number = STATE_EMPTY_CARD_SLOT
                card_color = STATE_EMPTY_CARD_SLOT
                if j < len(player_cards):
                    card_number = player_cards[j].number
                    card_color = player_cards[j].color.value
                state[index] = card_number
                state[index + 1] = card_color
                index += 2

        for player in hanabi_game._players:
            player_clues = hanabi_game.get_player_clues(player)
            for j in range(hanabi_game._initial_hand_size):
                clue_number = STATE_EMPTY_CARD_SLOT
                clue_color = STATE_EMPTY_CARD_SLOT
                if j < len(player_clues):
                    clue_number = player_clues[j].number
                    clue_number = clue_number if clue_number is not None else STATE_EMPTY_CLUE_SLOT
                    clue_color = player_clues[j].color
                    clue_color = clue_color.value if clue_color is not None else STATE_EMPTY_CLUE_SLOT
                state[index] = clue_number
                state[index + 1] = clue_color
                index += 2
        state[index] = hanabi_game.clues
        state[index + 1] = hanabi_game.bombs

        return state
