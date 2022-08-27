import random

from abc import ABC, abstractmethod
from typing import Tuple, Dict, List

from .models import Color, Card, Clue, Player
from .exceptions import NoClueEnough, InvalidCard, GameOver, CannotGiveClueToYourself


class HanabiInterface(ABC):
    @abstractmethod
    def __init__(self, cards_set: Tuple[Card], clues_set: Tuple[Clue], players: Tuple[Player], 
                 initial_hand_size: int, max_clues: int = 8, max_bombs: int = 3):
        raise NotImplementedError

    @abstractmethod
    def start(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def current_player(self) -> Player:
        raise NotImplementedError

    @property
    @abstractmethod
    def clues(self) -> int:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def bombs(self) -> int:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def points(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def piles(self) -> Dict[Color, List[Card]]:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def discard_zone(self) -> Dict[Card, int]:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def max_points(self) -> int:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def total_deck_cards(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def has_clues(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def has_cards_in_deck(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def has_won(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_alive(self) -> bool:
        raise NotImplementedError

    @staticmethod
    def _alive_game(func):
        def _wrapper(self, *args, **kwargs):
            if not self.is_alive():
                raise GameOver()
            return func(self, *args, **kwargs)
        return _wrapper
    
    @_alive_game
    @abstractmethod
    def give_clue(self, to: Player, clue: Clue):
        raise NotImplementedError
    
    @_alive_game
    @abstractmethod
    def discard_card(self, index: int) -> Card:
        raise NotImplementedError

    @_alive_game
    @abstractmethod
    def play_card(self, index: int) -> Card:
        raise NotImplementedError

    @abstractmethod
    def get_player_cards(self, player: Player) -> List[Card]:
        raise NotImplementedError

    @abstractmethod
    def get_player_clues(self, player: Player) -> List[Clue]:
        raise NotImplementedError

    @abstractmethod
    def _add_card_in_pile(self, card: Card):
        raise NotImplementedError

    @abstractmethod
    def _pop_player_card(self, player: Player, card: Card):
        raise NotImplementedError

    @abstractmethod
    def _player_draw(self, player: Player) -> Card:
        raise NotImplementedError

    @abstractmethod
    def _draw(self) -> Card:
        raise NotImplementedError

    @abstractmethod
    def _next_turn(self):
        raise NotImplementedError
