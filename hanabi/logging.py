import textwrap

from colorama import Fore, Style

from .game import Hanabi
from .models import Player, Color


def hanabi_game_str(hanabi_game: Hanabi):
    state_str = '-' * 46 + '\n\n'

    deck = str(hanabi_game.total_deck_cards).zfill(2)
    turn = str(hanabi_game.turn).zfill(2)
    points = str(hanabi_game.points).zfill(2)
    state_str += f'Deck: {deck}' + ' ' * 29 + f'Turn: {turn}\n'
    state_str += f'Clues: {hanabi_game.clues}' + ' ' * 9
    state_str += f'Points: {points}' + ' ' * 10
    state_str += f'Bombs: {hanabi_game.bombs}\n\n'

    clues_text = dict()
    cards_text = dict()
    color_styles = {Color.BLUE: Fore.BLUE, Color.WHITE: Fore.WHITE, 
                    Color.YELLOW: Fore.YELLOW, Color.RED: Fore.RED,
                    Color.GREEN: Fore.GREEN} 

    for player, clues in hanabi_game._player_clues.items():
        clues_text[player] = []
        for clue in clues:
            text = str(clue.number) if clue.number is not None else '█'
            text += Style.RESET_ALL
            clues_text[player].append(color_styles.get(clue.color, Fore.LIGHTBLACK_EX) + text)
    
    for player, cards in hanabi_game._player_hands.items():
        cards_text[player] = []
        for card in cards:
            text = str(card.number) + Style.RESET_ALL
            cards_text[player].append(color_styles.get(card.color, Fore.LIGHTBLACK_EX) + text)
    
    for player in hanabi_game._players:
        current_turn = '*' if player is hanabi_game.current_player else ' ' 
        state_str += current_turn + ' ' + textwrap.shorten(f'Player {player.id}', width=9, placeholder=' ').ljust(9) + ' '
        state_str += '[' + ', '.join(cards_text[player]) + ']' + ' ' * 3
        state_str += '[' + ', '.join(clues_text[player]) + ']' + '\n'

    state_str += Style.RESET_ALL + '\n'
    state_str += 'Played Zone' + ' ' * 17 + 'Discard Zone\n'

    discard_zone = [[] for _ in range(len(hanabi_game._colors))]

    for card in hanabi_game._discard_set:
        color_index = hanabi_game._colors.index(card.color)
        discard_zone[color_index].append(color_styles[card.color] + str(card.number))

    max_discard_colors_list = max(discard_zone, key=lambda d: len(d))
    total_zone_lines = max(hanabi_game._unique_cards_by_color, len(max_discard_colors_list))

    discard_zone_lines = [[' '] * len(hanabi_game._colors) for _ in range(total_zone_lines)]

    for color_index, card_texts in enumerate(discard_zone):
        for card_index, card_text in enumerate(card_texts):
            discard_zone_lines[card_index][color_index] = card_text

    played_zone_lines = [[' '] * len(hanabi_game._colors) for _ in range(total_zone_lines)]

    for color, cards in hanabi_game._piles.items():
        color_index = hanabi_game._colors.index(color)
        for card in cards:
            played_zone_lines[card.number - 1][color_index] = color_styles[color] + str(card.number)
    
    played_zone_lines = ['   '.join(line) for line in played_zone_lines]
    discard_zone_lines = ['   '.join(line) for line in discard_zone_lines]

    for played_zone_line, discard_zone_line in zip(played_zone_lines, discard_zone_lines):
        state_str += played_zone_line + ' ' * 11 + discard_zone_line + '\n'
    
    state_str += Style.RESET_ALL + '\n'
    state_str += '-' * 46 + '\n\n'

    return state_str
