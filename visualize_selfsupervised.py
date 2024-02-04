import csv
import numpy as np
from torch.utils.data import DataLoader
import torch
from chess_royale_environment import Game
from typing import List
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from chess_royale_ai.chess_royale_mcts import game_states_to_inputs
from chess_royale_ai.transformer_network import Chessformer
from selfsupervised import * 

if __name__ == "__main__":

    string_notation_file_name = "2_player_9_9"
    board_size = (9,9)
    batch_size = 32
    update_period = 100
    load_path = "./weights/self_supervised"

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
    print("Red: Only predicted, Blue: Only true, Magenta: Correct")
    with torch.no_grad():
        for batch_index, (inputs, selectables, targetables, selected) in enumerate(loader):

            inputs = [i.to(device) for i in inputs]
            selectables = selectables.to(device)
            targetables = targetables.to(device)
            selected = selected.to(device)

            outputs = chessformer(inputs, selection=selected)

            s_loss = select_loss(torch.sigmoid(outputs[0]), selectables)
            t_loss = target_loss(torch.sigmoid(outputs[1]), targetables)
            loss = s_loss + t_loss

            selectables = selectables
            targetables = targetables

            p_selectables = torch.sigmoid(outputs[0])
            p_targetables = torch.sigmoid(outputs[1])

            sels = torch.stack((p_selectables, torch.zeros_like(p_selectables), selectables)).permute((1,2,3,0)).cpu().numpy()
            targs = torch.stack((p_targetables, torch.zeros_like(p_targetables), targetables)).permute((1,2,3,0)).cpu().numpy()
            
            # for i in range(batch_size):
            #     plt.subplot(4,int(batch_size/4),i+1)
            #     plt.imshow(np.stack(sels[i]))

            for i in range(batch_size):
                plt.subplot(4,int(batch_size/4),i+1)
                plt.imshow(np.stack(targs[i]))

            # plt.draw()
            # plt.pause(1)
            # input("Press enter to continue")
            plt.show()