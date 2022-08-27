import random

from typing import Tuple, Dict, List

from .interface import HanabiInterface
from .models import Color, Card, Clue, Player
from .exceptions import NoClueEnough, InvalidCard, GameOver, CannotGiveClueToYourself


class Hanabi(HanabiInterface):
    def __init__(self, cards_set: Tuple[Card], clues_set: Tuple[Clue], players: Tuple[Player], 
                 initial_hand_size: int, max_clues: int = 8, max_bombs: int = 3):
                self._cards_set = cards_set
                self._clues_set = clues_set
                self._players = players
                self._max_clues = max_clues
                self._max_bombs = max_bombs
                self._initial_hand_size = initial_hand_size
                self._colors = set(card.color for card in self._cards_set)
                self._unique_cards_by_color = 5
                self._max_points = len(self._colors) * self._unique_cards_by_color

                self._deck = []
                self._discard_zone: Dict[Card, int] = dict()
                self._piles: Dict[Color, List[Card]] = dict()
                self._player_hands: Dict[Player, List[Card]] = dict()
                self._player_clues: Dict[Player, List[Clue]] = dict()
                self._current_player = 0
                self._clues = self._max_clues
                self._bombs = 0
                self._end_deck_countdown = 0

    def start(self):
        self._deck = random.sample(self._cards_set, k=len(self._cards_set))
        self._discard_zone = dict()
        self._piles = {color: [] for color in self._colors}
        self._player_hands = {player: [] for player in self._players}
        self._player_clues = {player: [] for player in self._players}

        for player in self._players:
            for _ in range(self._initial_hand_size):
                self._player_draw(player)

        self._current_player = 0
        self._clues = self._max_clues
        self._bombs = 0
        self._end_deck_countdown = len(self._players)

    @property
    def current_player(self) -> Player:
        return self._players[self._current_player]

    @property
    def clues(self) -> int:
        return self._clues
    
    @property
    def bombs(self) -> int:
        return self._bombs
    
    @property
    def points(self) -> int:
        return sum(len(pile) for pile in self._piles.values())

    @property
    def piles(self) -> Dict[Color, List[Card]]:
        return self._piles
    
    @property
    def discard_zone(self) -> Dict[Card, int]:
        return self._discard_zone
    
    @property
    def max_points(self) -> int:
        return self._max_points
    
    @property
    def total_deck_cards(self) -> int:
        return len(self._deck)

    def has_clues(self) -> bool:
        return self._clues > 0
    
    def has_cards_in_deck(self) -> bool:
        return self.total_deck_cards > 0
    
    def has_won(self) -> bool:
        return self.points == self._max_points

    def is_alive(self) -> bool:
        return self._bombs < self._max_bombs and (self._deck or self._end_deck_countdown >= 0) \
            and not self.has_won()

    @staticmethod
    def _alive_game(func):
        def _wrapper(self, *args, **kwargs):
            if not self.is_alive():
                raise GameOver()
            return func(self, *args, **kwargs)
        return _wrapper
    
    @_alive_game
    def give_clue(self, to: Player, clue: Clue):
        if not self.has_clues():
            raise NoClueEnough()
        
        if self.current_player is to:
            raise CannotGiveClueToYourself()

        for i in range(len(self._player_hands[to])):
            card = self._player_hands[to][i]
            if clue.color == card.color:
                self._player_clues[to][i].color = clue.color
            if clue.number == card.number:
                self._player_clues[to][i].number = clue.number

        self._clues -= 1
        self._next_turn()
    
    @_alive_game
    def discard_card(self, index: int) -> Card:
        card = self._player_hands[self.current_player][index]

        self._pop_player_card(self.current_player, card)
        
        self._discard_zone.setdefault(card, 0)
        self._discard_zone[card] += 1

        if self.has_cards_in_deck():
            self._player_draw(self.current_player)
        
        self._gain_clue()

        self._next_turn()

        return card

    @_alive_game
    def play_card(self, index: int) -> Card:
        card = self._player_hands[self.current_player][index]

        self._pop_player_card(self.current_player, card)

        if self._add_card_in_pile(card):
            if len(self._piles[card.color]) == self._unique_cards_by_color:
                self._gain_clue()
        else:
            self._bombs += 1

        if self.is_alive():
            if self.has_cards_in_deck():
                self._player_draw(self.current_player)
        
        self._next_turn()
        
        return card

    def get_player_cards(self, player: Player) -> List[Card]:
        return self._player_hands[player]
    
    def get_player_clues(self, player: Player) -> List[Clue]:
        return self._player_clues[player]

    def _add_card_in_pile(self, card: Card) -> bool:
        if len(self._piles[card.color]) < self._unique_cards_by_color \
                and card.number == len(self._piles[card.color]) + 1:
            self._piles[card.color].append(card)
            return True
        return False
    
    def _gain_clue(self):
        if self._clues < self._max_clues:
            self._clues += 1
    
    def _pop_player_card(self, player: Player, card: Card):
        try:
            card_index = self._player_hands[player].index(card)
        except ValueError:
            raise InvalidCard(card)
        else:
            self._player_clues[player].pop(card_index)
            self._player_hands[player].pop(card_index)

    def _player_draw(self, player: Player) -> Card:
        card = self._draw()
        self._player_hands[player].append(card)
        self._player_clues[player].append(Clue())
        return card

    def _draw(self) -> Card:
        card = self._deck.pop(0)
        return card

    def _next_turn(self):
        self._current_player = (self._current_player + 1) % len(self._players)

        if self.is_alive() and not self.has_cards_in_deck():
            self._end_deck_countdown -= 1
