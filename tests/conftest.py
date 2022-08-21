import pytest
import random

from hanabi.game import Hanabi
from hanabi.modes import NORMAL
from hanabi.models import Player

@pytest.fixture(scope='session')
def players():
    return [Player('A'), Player('B')]


@pytest.fixture
def hanabi_game(players):
    random.seed(1)
    initial_hand_size = 5
    hanabi = Hanabi(cards_set=NORMAL.cards, clues_set=NORMAL.clues, players=players, initial_hand_size=initial_hand_size)
    hanabi.start()
    return hanabi
