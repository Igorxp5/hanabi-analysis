import enum

from typing import Tuple
from dataclasses import dataclass


class Color(enum.Enum):
    RED = 0
    GREEN = 1
    BLUE = 2
    YELLOW = 3
    WHITE = 4
    MULTI = 5


@dataclass(frozen=True, eq=True, unsafe_hash=False)
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

    def __hash__(self):
        return object.__hash__(self)
    
    def immutable_hash(self):
        return hash((self.number, self.color))


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
