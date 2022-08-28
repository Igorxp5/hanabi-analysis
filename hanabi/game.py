import enum
import random
import dataclasses

from typing import Tuple, Dict, List, Set

from .interface import HanabiInterface
from .models import Color, Card, Clue, Player
from .exceptions import NoClueEnough, InvalidCard, GameOver, CannotGiveClueToYourself


class CardState(enum.Enum):
    NOT_PLAYED = 0
    PLAYED = 1
    DISCARDED = -1


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
        self._colors = [color for color in list(Color) if color in self._colors]  # get Color always in same order
        self._unique_cards_by_color = 5
        self._max_points = len(self._colors) * self._unique_cards_by_color

        self._deck: List[Card] = []
        self._cards_state: Dict[Card, CardState] = dict()
        self._discard_set: Set[Card] = set()
        self._discard_count: Dict[Card, int] = dict()
        self._piles: Dict[Color, List[Card]] = dict()
        self._player_hands: Dict[Player, List[Card]] = dict()
        self._player_clues: Dict[Player, List[Clue]] = dict()
        self._current_player = 0
        self._clues = self._max_clues
        self._bombs = 0
        self._turn = 0
        self._end_deck_countdown = 0

        self.start()

    def start(self):
        self._deck = random.sample(self._cards_set, k=len(self._cards_set))
        self._cards_state = {card: CardState.NOT_PLAYED for card in self._cards_set}
        self._discard_count = dict()
        self._discard_set = set()
        self._piles = {color: [] for color in self._colors}
        self._player_hands = {player: [] for player in self._players}
        self._player_clues = {player: [] for player in self._players}

        for player in self._players:
            for _ in range(self._initial_hand_size):
                self._player_draw(player)

        self._current_player = 0
        self._clues = self._max_clues
        self._bombs = 0
        self._turn = 1
        self._end_deck_countdown = len(self._players)

    @property
    def current_player(self) -> Player:
        return self._players[self._current_player]
    
    @property
    def other_players(self) -> Player:
        other_players = list(self._players)
        other_players.remove(self.current_player)
        return other_players

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
    def turn(self) -> int:
        return self._turn

    @property
    def piles(self) -> Dict[Color, List[Card]]:
        return self._piles
    
    @property
    def discard_set(self) -> Set[Card]:
        return self._discard_set

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
    def discard_card(self, index: int) -> Card:
        card = self._player_hands[self.current_player][index]

        self._pop_player_card(self.current_player, card)
        
        card_hash = card.immutable_hash()
        self._discard_count.setdefault(card_hash, 0)
        self._discard_count[card_hash] += 1
        self._discard_set.add(card)
        self._cards_state[card] = CardState.DISCARDED

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
            self._cards_state[card] = CardState.PLAYED
        else:
            self._bombs += 1
            self._cards_state[card] = CardState.DISCARDED

        if self.is_alive():
            if self.has_cards_in_deck():
                self._player_draw(self.current_player)
        
        self._next_turn()
        
        return card

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

    def get_player_cards(self, player: Player) -> List[Card]:
        return self._player_hands[player]
    
    def get_player_clues(self, player: Player) -> List[Clue]:
        return self._player_clues[player]
    
    def copy(self):
        hanabi_copy = Hanabi(self._cards_set, self._clues_set, self._players, self._initial_hand_size,
                        self._max_clues, self._max_bombs)
        hanabi_copy._deck = self._deck.copy()
        hanabi_copy._cards_state = self._cards_state.copy()
        hanabi_copy._discard_set = self._discard_set.copy()
        hanabi_copy._discard_count = self._discard_count.copy()

        hanabi_copy._piles = {color: cards.copy() for color, cards in self._piles.items()}
        hanabi_copy._player_hands = {player: cards.copy() for player, cards in self._player_hands.items()}
        
        for player in self._player_clues:
            for clue in self._player_clues[player]:
                hanabi_copy._player_clues[player].append(dataclasses.replace(clue))

        hanabi_copy._current_player = self._current_player
        hanabi_copy._clues = self._clues
        hanabi_copy._bombs = self._bombs
        hanabi_copy._turn = self._turn
        hanabi_copy._end_deck_countdown = self._end_deck_countdown

        return hanabi_copy

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
        
        self._turn += 1
