import logging

from typing import Callable, Tuple, Iterable
from functools import partial

import numpy as np

import tensorflow as tf

from hanabi.game import CardState, Hanabi
from hanabi.models import Player, Card, Clue

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

        logging.debug(f'Game state shape: {self._state_shape}')

        model_input = tf.keras.layers.Input(shape=self._state_shape)

        hidden_layer_1 = tf.keras.layers.Dense(units=1024, activation='relu')(model_input)
        hidden_layer_2 = tf.keras.layers.Dense(units=512, activation='relu')(hidden_layer_1)
        hidden_layer_3 = tf.keras.layers.Dense(units=256, activation='relu')(hidden_layer_2)

        self._policy = tf.keras.layers.Dense(self._actions_shape[0], activation='softmax', name='policy')(hidden_layer_3)
        self._state_value = tf.keras.layers.Dense(1, activation='tanh', name='state_value')(hidden_layer_3)

        self._model = tf.keras.models.Model(inputs=model_input, outputs=[self._policy, self._state_value])
        self._model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=tf.keras.optimizers.Adam(learning_rate))

        self.batch_size = batch_size
        self.epochs = epochs
    
    @property
    def metrics(self):
        return self._model.metrics

    def train(self, examples: Iterable[TrainingExample], fit_callback=None):
        states, policies, rewards = list(zip(*examples))
        states = np.asarray(states)
        policies = np.asarray(policies)
        rewards = np.asarray(rewards)

        callbacks = [fit_callback] if fit_callback else []
        self._model.fit(x=states, y=[policies, rewards], batch_size=self.batch_size,
                        epochs=self.epochs, verbose=0, callbacks=callbacks)

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
        
        deck_cards = np.zeros(shape=(len(hanabi_game._cards_set), 3))  # NOT_PLAYED, PLAYED, DISCARDED
        
        card_states = list(CardState)
        for i, card in enumerate(hanabi_game._cards_set):
            state_index = card_states.index(hanabi_game._cards_state[card])
            deck_cards[i][state_index] = 1

        total_other_players = len(hanabi_game._players) - 1
        other_players_cards = np.zeros(shape=(total_other_players, hanabi_game._initial_hand_size,
                                              hanabi_game._unique_cards_by_color, len(hanabi_game._colors), 1))
        
        for p, player in enumerate(hanabi_game.other_players):
            player_cards = hanabi_game.get_player_cards(player)
            for h, card in enumerate(player_cards):
                card_number_index = card.number - 1
                color_index = hanabi_game._colors.index(card.color)
                other_players_cards[p][h][card_number_index][color_index] = 1

        players_clues = np.zeros(shape=(len(hanabi_game._players), hanabi_game._initial_hand_size,
                                        hanabi_game._unique_cards_by_color, len(hanabi_game._colors), 1))
        
        for p, player in enumerate(hanabi_game._players):
            player_clues = hanabi_game.get_player_clues(player)
            for h, clue in enumerate(player_clues):
                players_clues[p][h] = 1  # There's a card in the current slot
                if clue.number is not None:
                    clue_number_index = clue.number - 1
                    players_clues[p][h][:clue_number_index] = 0
                    players_clues[p][h][clue_number_index + 1:] = 0
                if clue.color is not None:
                    clue_color_index = hanabi_game._colors.index(card.color)
                    players_clues[p][h][:][:clue_color_index] = 0
                    players_clues[p][h][:][clue_color_index + 1:] = 0
        
        current_clues_bombs = np.array([hanabi_game.clues, hanabi_game.bombs])

        return np.concatenate((deck_cards, other_players_cards, players_clues, current_clues_bombs), axis=None)
