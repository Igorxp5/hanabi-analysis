import os
import sys
import random
import shutil
import pathlib
import logging
import argparse
import datetime
import statistics

from typing import List, Tuple

import colorama
import numpy as np
import tensorflow as tf

from mcts import HanabiMcts
from agent import HanabiAgent, TrainingExample

from hanabi.game import Hanabi
from hanabi.models import Player
from hanabi.modes import NORMAL
from hanabi.exceptions import NoClueEnough
from hanabi.logging import hanabi_game_str

LOGS_DIR = pathlib.Path('./logs/').absolute()


def execute_episode(hanabi_game: Hanabi, agent: HanabiAgent, degree_exploration=1,
                    num_mcts_simulations=25, e_greedy=0.9) -> List[TrainingExample]:
    logging.info('Starting episode')

    mcts = HanabiMcts(agent, degree_exploration)

    state_policies = []
    while hanabi_game.is_alive():
        logging.debug(f'\n\n{hanabi_game_str(hanabi_game)}')

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
    results: List[Tuple[int, int]] = [] 
    
    logging.info(f'Agent evaluation has started')
    players = [Player(f'{i}') for i in range(agent._total_players)]
    for i in range(1, num_games + 1):
        logging.info(f'Game {i}/{num_games} has started')
        hanabi_game = Hanabi(agent._cards_set, agent._clues_set, players, agent._initial_hand_size)

        selected_actions = []
        while hanabi_game.is_alive():
            logging.debug(f'\n\n{hanabi_game_str(hanabi_game)}')

            policy = agent.get_policy(hanabi_game)

            # Discard a card if the action is invalid
            try:
                agent.act(hanabi_game)
            except NoClueEnough:
                hanabi_game.discard_card(0)

            selected_actions.append(np.argmax(policy))

        logging.info(f'Game {i}/{num_games} has finished')

        results.append((hanabi_game.points, hanabi_game.bombs, selected_actions))

        logging.info(f'Game {i}/{num_games} result: {hanabi_game.points} points and {hanabi_game.bombs} bombs')

        total_wins += 1 if hanabi_game.points == hanabi_game.max_points else 0
    
    logging.info(f'Agent evaluation has finished, win rate: {total_wins * 100 / num_games}%')
    return total_wins / num_games, results


def set_seed(seed: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input-model', default=None, help='Weights file in h5 format to start the training')
    parser.add_argument('-o', '--output-model', default='./model.h5', help='File path where to save the agent weights')
    parser.add_argument('--delete-logs', action='store_true', default=False, help='Delete tensor board logs before starting to train')
    parser.add_argument('--use-same-deck', action='store_true', default=False, help='Traing the agent using always the same deck')
    parser.add_argument('--seed', default=1, help='Random seed')

    args = parser.parse_args()

    colorama.init()

    logging.basicConfig(level=logging.DEBUG)

    set_seed(args.seed)

    game_mode = NORMAL
    total_players = 2
    initial_hand_size = 5
    learning_rate = 0.01
    batch_size = 8
    training_epochs = 30
    degree_exploration = 1
    num_mcts_simulations = 120
    num_evaluation_games = 10
    e_greedy = 0.85

    players = [Player(f'{i}') for i in range(total_players)]
    agent = HanabiAgent(game_mode.cards, game_mode.clues, total_players, initial_hand_size, learning_rate, batch_size, training_epochs)

    if args.input_model:
        agent.load_model(args.input_model)
    
    if args.delete_logs:
        shutil.rmtree(LOGS_DIR, ignore_errors=True)
    
    current_execution_logs = LOGS_DIR / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    evaluation_summary = tf.summary.create_file_writer(str(current_execution_logs / 'evaluation'))
    execution_summary = tf.summary.create_file_writer(str(current_execution_logs / 'execution'))
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(current_execution_logs / 'training'))

    try:
        logging.info('Starting training')
        episode = 0
        while True:
            logging.info(f'Starting episode {episode + 1}')

            if args.use_same_deck:
                set_seed(args.seed)

            hanabi_game = Hanabi(cards_set=game_mode.cards, clues_set=game_mode.clues, players=players, initial_hand_size=initial_hand_size)
            training_examples = execute_episode(hanabi_game, agent, degree_exploration, num_mcts_simulations, e_greedy)

            with execution_summary.as_default():
                tf.summary.scalar('mcts_points', hanabi_game.points, step=episode,
                                  description='Points reached by the episode run (MCTS)')
                tf.summary.scalar('mcts_bombs', hanabi_game.bombs, step=episode,
                                  description='Bombs get by the episode run (MCTS)')

                selected_actions = np.array([np.argmax(p) for _, p, _ in training_examples], dtype='int64')
                tf.summary.histogram('mcts_action', selected_actions, step=episode,
                                     description='Actions performed during episode run (MCTS)')

            logging.info('Traning the model with the episode examples')
            agent.train(training_examples, fit_callback=tensorboard_callback)

            logging.info('Training has finished')

            logging.info('Starting agent evaluation phase')
            if args.use_same_deck:
                logging.warning(f'All agent evaluation phases are using the same {num_evaluation_games} decks')
                logging.warning('The first agent evaluation will use the same deck used in training phase')
                set_seed(args.seed)
            
            win_rate, results = evaluate_agent(agent, num_evaluation_games)

            max_points, _, _ = max(results, key=lambda result: result[0])
            _, max_bombs, _ = max(results, key=lambda result: result[1])
            mean_points = statistics.mean(points for points, _, _ in results)

            with evaluation_summary.as_default():
                tf.summary.scalar('win_rate', win_rate, step=episode,
                                  description='Win rate of the model during the agent evalution')
                tf.summary.scalar('max_points', max_points, step=episode,
                                  description='Max points in game of the model during the agent evalution')
                tf.summary.scalar('max_bombs', max_bombs, step=episode,
                                  description='Max bombs in game of the model during the agent evalution')
                tf.summary.scalar('mean_points', mean_points, step=episode,
                                  description='Mean points in game of the model during the agent evalution')

                selected_actions = np.array([a for _, _, actions in results for a in actions], dtype='int64')
                tf.summary.histogram('selected_actions', selected_actions, step=episode,
                                     description='Actions performed during the agent evaluation')

            logging.info('Saving model')
            agent.export_model(args.output_model)

            episode += 1
    except KeyboardInterrupt:
        logging.warning('Stopping training! All data generated in this epoch will be discarded!')
        sys.exit(0)

if __name__ == '__main__':
    main()
