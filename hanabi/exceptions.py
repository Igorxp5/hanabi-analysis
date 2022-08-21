class NoClueEnough(RuntimeError):
    pass


class InvalidCard(RuntimeError):
    pass


class GameOver(RuntimeError):
    pass


class CannotGiveClueToYourself(RuntimeError):
    pass