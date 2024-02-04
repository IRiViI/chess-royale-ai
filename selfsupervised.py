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


def get_string_notations(file_path):
    string_notations = []
    with open(file_path, "r", encoding='UTF8', newline='') as file_object:
        reader = csv.reader(file_object)
        # Discard the header
        next(reader)
        # append string_notations
        for row in reader:
            string_notations.append(row[1])
    return string_notations

class CollateWrapper():

    def __init__(self):
        pass

    def __call__(self, string_notations):
        games = [Game.from_string_notation(string_notation) for string_notation in string_notations]
        game_states = [game.state() for game in games]
        inputs = game_states_to_inputs(game_states=game_states)
        selectables, targetables, selected = games_to_selectable_and_single_targetable(games)
        return inputs, selectables, targetables, selected

def games_to_selectable_and_single_targetable(games: List[Game]):
    batch_size = len(games)
    size = games[0].board.size
    selectables = torch.zeros((batch_size, *size), dtype=torch.float32)
    targetables = torch.zeros((batch_size, *size), dtype=torch.float32)
    selected = torch.zeros((batch_size, 2), dtype=torch.int64)
    for game_index, game in enumerate(games):
        selectables[game_index], targetables[game_index], selected[game_index] = moves_to_selectable_and_single_targetable(game.get_legal_moves(),size)
    return selectables, targetables, selected

def moves_to_selectable_and_single_targetable(moves, size):
    selectables = torch.zeros(size, dtype=torch.float32)
    targetables = torch.zeros(size, dtype=torch.float32)
    unique_froms = np.unique(moves[:,0,:], axis=0)
    selected = unique_froms[np.random.randint(len(unique_froms))]
    for f, t in moves:
        selectables[f[0], f[1]] = 1
        if np.array_equal(f, selected):
            targetables[t[0], t[1]] = 1
    return selectables, targetables, torch.tensor(selected, dtype=torch.int64)

if __name__ == "__main__":

    string_notation_file_name = "2_player_9_9"
    board_size = (9,9)
    batch_size = 16
    update_period = 100
    load_path = "./weights/self_supervised"
    save_path = "./weights/self_supervised"

    # Model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    chessformer = Chessformer(board_shape=board_size)
    # chessformer = Chessformer(board_shape=board_size, embed_dim=16, num_heads=4, shared_encoder_layers=2)
    chessformer.load_state_dict(torch.load(load_path))
    # chessformer.weakly_load_state_dict(torch.load(load_path))
    chessformer.to(device)

    select_loss = torch.nn.BCELoss()
    target_loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(chessformer.parameters(), lr=1e-5)

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

    # Train
    for _ in range(100):
        progress_bar = tqdm(range(len(loader)))
        running_loss = 0.0
        
        for batch_index, (inputs, selectables, targetables, selected) in enumerate(loader):

            optimizer.zero_grad()

            inputs = [i.to(device) for i in inputs]
            selectables = selectables.to(device)
            targetables = targetables.to(device)
            selected = selected.to(device)

            outputs = chessformer(inputs, selection=selected)

            s_loss = select_loss(torch.sigmoid(outputs[0]), selectables)
            t_loss = target_loss(torch.sigmoid(outputs[1]), targetables)
            loss = s_loss + t_loss
        
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_index % update_period == update_period-1:
            
                progress_bar.update(update_period)
                progress_bar.set_description(str(running_loss / update_period))

                running_loss = 0.0

            if batch_index % 10_000 == 0:
                torch.save(chessformer.state_dict(), save_path)

