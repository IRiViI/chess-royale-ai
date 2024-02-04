
import csv
from chess_royale_environment import Game
from chess_royale_environment.boards import RandomBoard

if __name__ == "__main__":

    num_of_players = 2
    num_games = 100_000
    board_size = (9,9)
    max_num_steps = 16
    file_name = "2_player"

    with open(f"string_notations/{file_name}.csv", "w", encoding='UTF8', newline='') as file_object:
        writer = csv.writer(file_object)
        writer.writerow(['game_index', "string_notation"])
        for game_index in range(num_games):

            if game_index % 100 == 0:
                print(f"{game_index}/{num_games}")

            board = RandomBoard(num_players=num_of_players, size=board_size)
            game = Game(board=board)
            game.randomize_pieces()

            for _ in range(max_num_steps):
                move = game.get_random_legal_move()
                if move is not None:
                    f, t = move
                    game.step(f, t)
                else:
                    break
                string_notation = game.string_notation()
                # print(string_notation)
                writer.writerow([game_index, string_notation])

            