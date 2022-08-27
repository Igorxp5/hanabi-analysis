from typing import List

import pytest

from hanabi.game import Hanabi
from hanabi.modes import NORMAL
from hanabi.models import Player, Card, Clue, Color
from hanabi.exceptions import CannotGiveClueToYourself, GameOver


def test_game_start(hanabi_game: Hanabi, players: List[Player]):
    """The players should receive the initial cards when the game starts"""
    assert hanabi_game.clues == 8
    assert hanabi_game.bombs == 0

    for player in players:
        assert all(isinstance(card, Card) for card in hanabi_game.get_player_cards(player))
        assert len(hanabi_game.get_player_cards(player)) == 5


def test_game_play_card(hanabi_game: Hanabi, players: List[Player]):
    """
    Player should be able to play a card, if the card can be part of the pile
    no bomb point is get, otherwise, the bomb point is increased
    """
    card = hanabi_game.play_card(0)

    assert hanabi_game.get_player_cards(players[0])[1] is not card
    assert len(hanabi_game.get_player_cards(players[0])) == 5
    assert hanabi_game.points == 1
    assert len(hanabi_game.piles[Color.BLUE]) == 1
    assert hanabi_game.bombs == 0

    assert hanabi_game.piles[Color.BLUE][0] is card

    card = hanabi_game.play_card(0)

    assert hanabi_game.get_player_cards(players[1])[0] is not card
    assert len(hanabi_game.get_player_cards(players[1])) == 5
    assert hanabi_game.points == 1
    assert hanabi_game.bombs == 1


def test_game_over_by_bombs(hanabi_game: Hanabi, players: List[Player]):
    """The game cannot continue if the bomb points reach max bombs"""
    hanabi_game.play_card(3)
    hanabi_game.play_card(0)
    hanabi_game.play_card(1)

    assert hanabi_game.bombs == 3
    assert not hanabi_game.is_alive()

    with pytest.raises(GameOver):
        hanabi_game.play_card(1)
    
    with pytest.raises(GameOver):
        hanabi_game.discard_card(1)
    
    with pytest.raises(GameOver):
        hanabi_game.give_clue(players[0], Clue(number=1))


def test_game_over_by_deck_empty(hanabi_game: Hanabi, players: List[Player]):
    """
    The game should over when the deck empty. When the last card is get by some player,
    it starts the last round.
    """
    for _ in range(40):
        hanabi_game.discard_card(0)
    
    assert not hanabi_game.has_cards_in_deck()

    assert all(len(hanabi_game.get_player_cards(player)) == 5 for player in players)

    hanabi_game.discard_card(0)

    assert len(hanabi_game.get_player_cards(players[0])) == 4

    hanabi_game.discard_card(0)

    assert len(hanabi_game.get_player_cards(players[1])) == 4

    with pytest.raises(GameOver):
        hanabi_game.discard_card(0)


def test_game_over_by_won(hanabi_game: Hanabi, players: List[Player]):
    """The game should not continue if the players have reached the max points"""
    for color in hanabi_game._colors:
        for i in range(1, 6):
            hanabi_game.piles[color].append(Card(number=i, color=color))
    
    assert not hanabi_game.is_alive()
    assert hanabi_game.has_won()

    with pytest.raises(GameOver):
        hanabi_game.play_card(0)


def test_extra_clue_by_playing_last_pile_card(hanabi_game: Hanabi):
    """The players should gain extra clue by play the last card of a color in the pile"""
    hanabi_game._clues = 7

    for i in range(1, 5):
        hanabi_game.piles[Color.RED].append(Card(number=i, color=Color.RED))
    
    for _ in range(13):
        hanabi_game.discard_card(0)
    
    hanabi_game.play_card(4)

    assert len(hanabi_game.piles[Color.RED])
    assert hanabi_game.clues == 8


def test_gain_bomb_by_playing_in_full_pile(hanabi_game: Hanabi):
    """The players should get a bomb point by playing a card into a full pile"""
    for i in range(1, 6):
        hanabi_game.piles[Color.RED].append(Card(number=i, color=Color.RED))
    
    for _ in range(13):
        hanabi_game.discard_card(0)
    
    hanabi_game.play_card(4)

    assert hanabi_game.bombs == 1


def test_game_next_turn(hanabi_game: Hanabi, players: List[Player]):
    """
    Any of the actions (give clue, discard a card, play a card) 
    should get the turn to the next player.
    """
    assert hanabi_game.current_player == players[0]

    hanabi_game.play_card(0)

    assert hanabi_game.current_player == players[1]

    hanabi_game.discard_card(0)

    assert hanabi_game.current_player == players[0]

    hanabi_game.give_clue(players[1], Clue(number=3))

    assert hanabi_game.current_player == players[1]


def test_give_clue(hanabi_game: Hanabi, players: List[Player]):
    """
    Player can give a clue to another player about number or color of their cards.
    The players should loose a clue point to do that.
    """
    player_a, player_b = players
    hanabi_game.give_clue(player_b, Clue(number=3))

    assert hanabi_game.clues == 7
    
    player_clues = hanabi_game.get_player_clues(player_b)
    assert all(player_clues[i].number == 3 for i in [0, 1, 2])
    assert all(player_clues[i].color is None for i in [0, 1, 2])
    assert all(player_clues[i].number is None for i in [3, 4])
    assert all(player_clues[i].color is None for i in [3, 4])

    with pytest.raises(CannotGiveClueToYourself):
        hanabi_game.give_clue(player_b, Clue(color=Color.RED))
    
    hanabi_game.discard_card(4)

    hanabi_game.give_clue(player_b, Clue(color=Color.YELLOW))

    assert player_clues[0].number == 3 and player_clues[0].color is Color.YELLOW
    assert player_clues[1].number == 3 and player_clues[1].color is None
    assert player_clues[2].number == 3 and player_clues[2].color is None
    assert player_clues[3].number is None and player_clues[3].color is Color.YELLOW
    assert player_clues[4].number is None and player_clues[4].color is None

    assert hanabi_game.clues == 7


def test_discard_card(hanabi_game: Hanabi):
    """
    Player who discard a card shuold get a new one and gain one clue for the team.
    If they reach the max clues they shouldn't not get the clue.
    """
    player = hanabi_game.current_player
    player_cards = hanabi_game.get_player_cards(player)
    discarded_card = hanabi_game.discard_card(0)
    assert hanabi_game.clues == 8
    assert len(player_cards) == 5
    assert player_cards[0] is not discarded_card
    assert discarded_card in hanabi_game.discard_zone

    hanabi_game.give_clue(player, Clue(number=3))

    player_cards = hanabi_game.get_player_cards(player)
    discarded_card = hanabi_game.discard_card(0)
    assert hanabi_game.clues == 8
    assert len(player_cards) == 5
    assert player_cards[0] is not discarded_card
    assert discarded_card in hanabi_game.discard_zone
