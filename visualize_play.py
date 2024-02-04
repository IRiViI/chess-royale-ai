import csv
import numpy as np
from torch.utils.data import DataLoader
import torch
from chess_royale_environment import Game
from chess_royale_environment.pieces import PieceKind, piece_kind_to_string_value
from typing import List
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from chess_royale_ai.chess_royale_mcts import game_states_to_inputs
from chess_royale_ai.transformer_network import Chessformer
from selfsupervised import games_to_selectable_and_single_targetable, get_string_notations


class CollateWrapper():

    def __init__(self):
        pass

    def __call__(self, string_notations):
        games = [Game.from_string_notation(string_notation) for string_notation in string_notations]
        game_states = [game.state() for game in games]
        inputs = game_states_to_inputs(game_states=game_states)
        selectables, targetables, selected = games_to_selectable_and_single_targetable(games)
        return games, inputs, selectables, targetables, selected
        
if __name__ == "__main__":

    string_notation_file_name = "2_player_9_9"
    board_size = (9,9)
    batch_size = 16
    update_period = 300
    load_path = "./weights/self_supervised"

    board_len = np.prod(board_size)

    # Model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    chessformer = Chessformer(board_shape=board_size)
    chessformer.weakly_load_state_dict(torch.load(load_path))
    chessformer.to(device)

    select_loss = torch.nn.BCELoss()
    target_loss = torch.nn.BCELoss()

    # Data
    string_notations = get_string_notations(f"string_notations/{string_notation_file_name}.csv")

    collate_wrapper = CollateWrapper()

    loader = DataLoader(string_notations,
        batch_size=batch_size,
        collate_fn=collate_wrapper,
        num_workers=2,
        pin_memory=True,
        shuffle=True
    )

    # Eval
    plt.figure(figsize=(14,14))
    print("Green: predicted but not possible, Blue: possible but nope, turqouise: Best predicted action")
    with torch.no_grad():
        for batch_index, (games, inputs, selectables, targetables, selected) in enumerate(loader):

            inputs = [i.to(device) for i in inputs]
            selectables = selectables.to(device)
            targetables = targetables.to(device)
            selected = selected.to(device)

            outputs = chessformer(inputs, selection=selected)

            s_loss = select_loss(torch.sigmoid(outputs[0]), selectables)
            t_loss = target_loss(torch.sigmoid(outputs[1]), targetables)
            loss = s_loss + t_loss
            

            p_selectables = torch.softmax(outputs[0].view(-1,board_len), dim=-1).view(-1,*board_size)
            p_targetables = torch.softmax(outputs[1].view(-1,board_len), dim=-1).view(-1,*board_size)

            sels = torch.stack((torch.zeros_like(p_selectables), p_selectables, 0.3 * selectables)).permute((1,2,3,0)).cpu().numpy()
            targs = torch.stack((torch.zeros_like(p_targetables), p_targetables, 0.3 * targetables)).permute((1,2,3,0)).cpu().numpy()
            values = outputs[3].cpu().numpy()
            
            for i in range(batch_size):
                plt.subplot(4,int(batch_size/2),2*(i+1)-1)
                plt.imshow(np.stack(sels[i]))
                plt.title(str(values[i]))
                game: Game = games[i]
                for x in range(board_size[0]):
                    for y in range(board_size[1]):
                        value:int = game.board.pieces_matrix[x,y]
                        player = game.board.player_matrix[x,y]
                        walkable = game.board.walkable_matrix[x,y]
                        if value > 0:
                            str_value:str = piece_kind_to_string_value(PieceKind(value))
                            if player == 2:
                                str_value = str_value.capitalize()
                            plt.text(y-0.3, x+0.3, str_value, c="white")
                        if walkable == 0:
                            plt.text(y-0.3, x+0.3, "X", c="white")


                plt.subplot(4,int(batch_size/2),2*(i+1))
                plt.imshow(np.stack(targs[i]))
                for x in range(board_size[0]):
                    for y in range(board_size[1]):
                        value:int = game.board.pieces_matrix[x,y]
                        player = game.board.player_matrix[x,y]
                        walkable = game.board.walkable_matrix[x,y]
                        if value > 0:
                            str_value:str = piece_kind_to_string_value(PieceKind(value))
                            if player == 2:
                                str_value = str_value.capitalize()
                            plt.text(y-0.3, x+0.3, str_value, c="white")
                        if walkable == 0:
                            plt.text(y-0.3, x+0.3, "X", c="white")

            # plt.draw()
            # plt.pause(1)
            # input("Press enter to continue")
            plt.show()