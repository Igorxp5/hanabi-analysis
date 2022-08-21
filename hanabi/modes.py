from .models import Color, Card, Clue, GameMode

_NORMAL_COLORS = Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW, Color.WHITE
_NORMAL_NUMBERS = {1: 3, 2: 2, 3: 2, 4: 2, 5: 1}
_NORMAL_CARDS = [Card(n, color) for n, total in _NORMAL_NUMBERS.items() for color in _NORMAL_COLORS for _ in range(total)]
_NORMAL_CLUES = [Clue(color=color) for color in _NORMAL_COLORS] + [Clue(number=number) for number in _NORMAL_NUMBERS]

_HARD_CLUES = _NORMAL_CLUES + [Clue(color=Color.MULTI)]
_HARD_CARDS = _NORMAL_CARDS + [Card(n, Color.MULTI) for n in range(1, 6)]

NORMAL = GameMode(cards=tuple(_NORMAL_CARDS), clues=tuple(_NORMAL_CLUES))
HARD = GameMode(cards=tuple(_HARD_CARDS), clues=tuple(_HARD_CLUES))
VERY_HARD = GameMode(cards=HARD.cards, clues=NORMAL.clues)
