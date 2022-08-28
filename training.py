import sys
import random
import logging
import argparse
from tkinter.font import NORMAL

from typing import List, Tuple

import numpy as np

from mcts import HanabiMcts
from agent import HanabiAgent, TrainingExample

from hanabi.game import Hanabi
from hanabi.models import Player
from hanabi.modes import NORMAL
from hanabi.exceptions import NoClueEnough


def execute_episode(hanabi_game: Hanabi, agent: HanabiAgent, degree_exploration=1,
                    num_mcts_simulations=25, e_greedy=0.9) -> List[TrainingExample]:
    logging.info('Starting episode')

    mcts = HanabiMcts(agent, degree_exploration)

    state_policies = []
    while hanabi_game.is_alive():
        logging.debug(f'Current turn: {hanabi_game.turn}')
        logging.debug(f'Current points: {hanabi_game.points}')
        logging.debug(f'Current bombs: {hanabi_game.bombs}')
        logging.debug(f'Current total deck cards: {hanabi_game.total_deck_cards}')

        state = HanabiAgent.get_game_state(hanabi_game)

        logging.debug('Running MCTS for current state')
        for _ in range(num_mcts_simulations):
            mcts.simulate(hanabi_game)
        logging.debug('MCTS simulations has finished')

        assert state.tobytes() == HanabiAgent.get_game_state(hanabi_game).tobytes()

        policy = mcts.get_policy(hanabi_game)

        # Using e-greedy, otherwise we always will use MCTS choice, that is the best action.
        # So the model will not generalize for new situation even though Hanabi games are random.

        if random.random() > e_greedy:
            valid_actions = HanabiAgent.get_valid_actions(hanabi_game)
            random_action = np.random.choice(np.where(valid_actions == 1)[0])
            policy = np.zeros(policy.shape)
            policy[random_action] = 1
        
        HanabiAgent.get_policy_action_function(hanabi_game, policy)()

        state_policies.append((HanabiAgent.get_game_state(hanabi_game), policy))
    
    reward = mcts.get_state_value(hanabi_game)
    examples = []

    logging.info('Episode execution has finished')

    for state, policy in state_policies:
        examples.append((state, policy, reward))

    return examples

def evaluate_agent(agent: HanabiAgent, num_games: int) -> float:
    total_wins = 0
    rewards: List[Tuple[int, int]] = [] 
    
    logging.info(f'Agent evaluation has started')
    players = [Player(f'{i}') for i in range(agent._total_players)]
    for i in range(1, num_games + 1):
        logging.info(f'Game {i}/{num_games} has started')
        hanabi_game = Hanabi(agent._cards_set, agent._clues_set, players, agent._initial_hand_size)

        while hanabi_game.is_alive():
            # Discard a card if the action is invalid
            try:
                agent.act(hanabi_game)
            except NoClueEnough:
                hanabi_game.discard_card(0)

        logging.info(f'Game {i}/{num_games} has finished')

        rewards.append((hanabi_game.points, hanabi_game.bombs))

        logging.info(f'Game {i}/{num_games} result: {hanabi_game.points} points and {hanabi_game.bombs} bombs')

        total_wins += 1 if hanabi_game.points == hanabi_game.max_points else 0
    
    logging.info(f'Agent evaluation has finished, win rate: {total_wins * 100 / num_games}%')
    return total_wins / num_games, rewards


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input-model', default='./model.h5', help='Weights file in h5 format to start the training')
    parser.add_argument('-o', '--output-model', default='./model.h5', help='File path where to save the agent weights')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    random.seed(1)
    
    game_mode = NORMAL
    total_players = 2
    initial_hand_size = 5
    learning_rate = 0.001
    batch_size = 64
    epochs = 10
    degree_exploration = 1
    num_mcts_simulations = 25
    num_evaluation_games = 10
    e_greedy = 0.9

    players = [Player(f'{i}') for i in range(total_players)]
    agent = HanabiAgent(game_mode.cards, game_mode.clues, total_players, initial_hand_size, learning_rate, batch_size, epochs)

    if args.input_model:
        agent.load_model(args.input_model)

    try:
        logging.info('Starting training')
        episode = 0
        while True:
            episode += 1
            
            logging.info(f'Starting episode {episode}')
            hanabi_game = Hanabi(cards_set=game_mode.cards, clues_set=game_mode.clues, players=players, initial_hand_size=initial_hand_size)
            training_examples = execute_episode(hanabi_game, agent, degree_exploration, num_mcts_simulations, e_greedy)
            
            logging.info('Traning the model with the episode examples')
            agent.train(training_examples, 1)
            
            logging.info('Training has finished')
            evaluate_agent(agent, num_evaluation_games)
            
            logging.info('Saving model')
            agent.export_model(args.output_model)
    except KeyboardInterrupt:
        logging.warning('Stopping training! All data generated in this epoch will be discarded!')
        sys.exit(0)

if __name__ == '__main__':
    main()
