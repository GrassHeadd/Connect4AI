from pygame_simulator import GameControllerPygame, HumanPygameAgent
from connect_four import ConnectFour
import contest

class ZeroAgent(object):
    def __init__(self, player_id):
        pass
    def make_move(self, state):
        return 0

if __name__ == "__main__":
    board = ConnectFour()
    game = GameControllerPygame(board=board, agents=[contest.AIAgent(1), contest.AIAgent(2)])
    game.run()