import enum

from typing import Tuple
from dataclasses import dataclass


class Color(enum.Enum):
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'
    YELLOW = 'yellow'
    WHITE = 'white'
    MULTI = 'multi'


@dataclass(frozen=True)
class Card:
    number: int = None
    color: Color = None

    def __lt__(self, other):
        return self.number < other.number
    
    def __le__(self, other):
        return self.number <= other.number
    
    def __gt__(self, other):
        return self.number > other.number
    
    def __gt__(self, other):
        return self.number >= other.number


@dataclass
class Clue:
    number: int = None
    color: Color = None


@dataclass(frozen=True)
class GameMode:
    cards: Tuple[Card]
    clues: Tuple[Clue]


@dataclass(frozen=True)
class Player:
    id: str
