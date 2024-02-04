import torch
from torch import multiprocessing, nn
from torch.nn import functional as F
from positional_encodings import PositionalEncoding2D, DiagonalPositionalEncoding2D, FixEncoding
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from scipy.special import softmax



class PositionSelection(nn.Module):
    def __init__(self, board_shape, dim, **kwargs):
        super(PositionSelection, self).__init__(**kwargs)
        # Fixed configs
        self.board_shape = board_shape
        self.board_len = np.prod(board_shape)
        # Layers
        self.linear = nn.Linear(dim+1, dim)
        
    def forward(self, inputs):
        """
            inputs: logits
        """
        tensors, position = inputs
        x = F.one_hot(position, num_classes=self.board_len).view(-1,self.board_len,1)
        x = torch.cat((tensors, x), dim=-1)
        x = self.linear(x)
        return x

class Sampler(nn.Module):
    def __init__(self, **kwargs):
        super(Sampler, self).__init__(**kwargs)
        
    def forward(self, logits=None, probabilities=None):
        if logits is not None and probabilities is None:
            distributions = F.softmax(logits, dim=-1)
        elif logits is None and probabilities is not None:
            distributions = probabilities
        else: 
            raise ValueError("Provide logits or probablities not both")
        return torch.multinomial(distributions,1).view(-1,)


class TokenPositionPiecePlayerEmbedding(nn.Module):
    def __init__(self, position_shape, piece_dim, player_dim, embed_dim, diagonals=True, **kwargs):
        super(TokenPositionPiecePlayerEmbedding, self).__init__(**kwargs)
        input_dims = 0
        self.token_emb = nn.Embedding(piece_dim, embed_dim)
        input_dims += embed_dim
        self.pos_enc = FixEncoding(PositionalEncoding2D(embed_dim), position_shape)
        input_dims += embed_dim
        self.diagonals = diagonals
        if self.diagonals:
            self.dia_pos_enc = FixEncoding(DiagonalPositionalEncoding2D(embed_dim), position_shape)
            input_dims += embed_dim
        self.player_dim = player_dim
        input_dims += 2*self.player_dim
        self.linear = torch.nn.Linear(input_dims + 3, embed_dim) # +3 due to walkable, hill and hill_unhold
        
    def forward(self, inputs):
        pieces, players, current_player, walkable, hills, hill_unholds = inputs
        # print(",", pieces.device, players.device, current_player.device, self.pos_enc.device)
        token_emb = self.token_emb(pieces)
        players_emb = F.one_hot(players, num_classes=3)
        current_players_emb = F.one_hot(current_player, num_classes=3)
        walkable = torch.unsqueeze(walkable, -1)
        hills = torch.unsqueeze(hills, -1)
        hill_unholds = torch.unsqueeze(hill_unholds, -1)
        # pos_emb = self.pos_enc(pieces)
        if self.diagonals:
            parts = (token_emb, self.pos_enc(pieces), self.dia_pos_enc(pieces), players_emb, current_players_emb, walkable, hills, hill_unholds)
        else:
            parts = (token_emb, self.pos_enc(pieces), players_emb, current_players_emb, walkable, hills, hill_unholds)
        # for part in parts:
        #     print(part.device)
        x = torch.cat(parts, axis=-1)
        y = self.linear(x)
        return y


class Chessformer(nn.Module):
    def __init__(self, shared_encoder_layers=5, target_encoder_layers=1,
                board_shape=(8,8), num_pieces=6, embed_dim=64, num_heads=8, num_players=2,
                dropout=0.0,
                select_mode="best", diagonals=True, **kwargs):
        super(Chessformer, self).__init__(**kwargs)
        
        # Fixed configs
        self.board_shape = board_shape
        self.board_len = np.prod(self.board_shape)
        self.piece_dim = num_pieces + 1
        self.embed_dim = embed_dim
        self.player_dim = num_players + 1
        self.num_heads = num_heads
        self.shared_encoder_layers = shared_encoder_layers
        self.target_encoder_layers = target_encoder_layers
        self.diagonals = diagonals
        
        # Flexible settings
        self.select_mode = None
        self.set_select_mode(select_mode)
        
        # Embedding
        self.token_position_piece_player_embedding = TokenPositionPiecePlayerEmbedding(board_shape, self.piece_dim, self.player_dim, embed_dim, 
            diagonals=self.diagonals)
        
        # Shared processing
        self.shared_encoders = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.embed_dim, nhead=self.num_heads, batch_first=True, dropout=dropout), 
            num_layers=self.shared_encoder_layers)
        
        # Predict selectables
        self.predict_selectables_encoder = nn.TransformerEncoderLayer(self.embed_dim, nhead=self.num_heads, batch_first=True, dropout=dropout)
        self.predict_selectables_linear = nn.Linear(self.embed_dim, 1)
        self.predict_selectables_flatten = nn.Flatten()
        
        # Select position
        self.sampler = Sampler()
        self.selector = PositionSelection(self.board_shape, self.embed_dim)
        
        # Predict targetables
        self.predict_targetables_encoders = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.embed_dim, nhead=self.num_heads, batch_first=True, dropout=dropout), 
            num_layers=self.target_encoder_layers)
        self.predict_targetables_linear = nn.Linear(self.embed_dim, 1)
        self.predict_targetables_flatten = nn.Flatten()

        # Value network
        self.value_linear = nn.Linear(self.embed_dim * self.board_len, num_players)

    def forward(self, inputs, selection=None):
        # 1. Embed the piece, player and position information
        embeding = self._embed(inputs)
        
        # 2. Process this information
        shared_latent_space = self._process(embeding)
        
        # 3. Predict selectables
        predict_selectables_logits = self._selectables(shared_latent_space)
        
        # 4. Selection
        applied_selection, selected = self._select(shared_latent_space, 
            selectables_logits=predict_selectables_logits, selection=selection)
        
        # 5. Predict targetables
        predict_targetables_logits = self._targetables(applied_selection)

        # 6. Predict value
        value = self.value_linear(shared_latent_space.view(-1, self.embed_dim * self.board_len))
        
        return (
            predict_selectables_logits.view((-1, self.board_shape[0], self.board_shape[1])), 
            predict_targetables_logits.view((-1, self.board_shape[0], self.board_shape[1])), 
            selected , 
            value
        )

    def _move_policies_and_values(self, inputs, n_froms_tensor, r_froms_tensor):
        # 1. Embed the piece, player and position information
        embeding = self._embed(inputs)
        
        # 2a. Process this information
        shared_latent_space = self._process(embeding)
        
        # 2b. repeat the shared latens space
        shared_latent_space_repeats = torch.repeat_interleave(shared_latent_space, n_froms_tensor, dim=0)

        # 3. Predict selectables
        predict_selectables_logits = self._selectables(shared_latent_space)
        
        # 4. Selection
        applied_selection, _ = self._select(shared_latent_space_repeats, selection=r_froms_tensor)

        # 5. Predict targetables
        predict_targetables_logits = self._targetables(applied_selection)

        # 6. Predict value
        values = self.value_linear(shared_latent_space.view(-1, self.embed_dim * self.board_len))
        return predict_selectables_logits, predict_targetables_logits, values

    def moves_policies_and_values(self, inputs, moves, selection_filters=None):
        with torch.no_grad():
            # 0. Process the moves into model friendly values
            device = inputs[0].device

            # Pre-processing
            froms_indices, tos_indices, n_froms, r_froms = preprocess_for_move_polices_and_values(moves, self.board_shape)
            r_froms_tensor = torch.tensor(r_froms, dtype=torch.int64).to(device)
            n_froms_tensor = torch.tensor(n_froms, dtype=torch.int64).to(device)
            
            # Get policies and logits
            predict_selectables_logits, predict_targetables_logits, values = self._move_policies_and_values(inputs, n_froms_tensor, r_froms_tensor)
            
            batch_size = predict_selectables_logits.shape[0]

            # Apply selection filter
            if selection_filters is not None:
                predict_selectables_logits[selection_filters.view(batch_size, -1)==0] = 0
            predict_selectables_probs = torch.softmax(predict_selectables_logits, dim=-1)
            predict_targetables_probs = torch.softmax(predict_targetables_logits, dim=-1)

            # Post-processing 
            policies = postprocess_for_move_polices_and_values(
                predict_selectables_probs.cpu().numpy(), 
                predict_targetables_probs.cpu().numpy(), 
                n_froms, 
                r_froms, 
                froms_indices, 
                tos_indices)
              
            return policies, values.cpu().numpy()

    def set_select_mode(self, mode):
        if mode not in ["best", "sample"]:
            raise ValueError("Select mode must be 'best' or 'sample'")
        self.select_mode = mode

    # Main model methods
    def _embed(self, inputs):
        embeding = self.token_position_piece_player_embedding(inputs)
        embeding = embeding.view(-1,self.board_len,self.embed_dim) # Flatten from 2D to 1D
        return embeding

    def _process(self, embeding):
        return self.shared_encoders(embeding)

    def _selectables(self, inputs):
        predict_selectables = self.predict_selectables_encoder(inputs)
        predict_selectables = self.predict_selectables_linear(predict_selectables)
        predict_selectables_logits = self.predict_selectables_flatten(predict_selectables)
        return predict_selectables_logits

    def _select(self, inputs, selectables_logits=None, selection=None):
        if selection is not None:
            if selection.ndim == 1:
                pass
            elif selection.shape[1] == 2:
                selection = self.board_shape[1] * selection[:,0] + selection[:,1]
            else:
                raise ValueError("Selection type must be a tensor with indices or tensor with positions")
            selected = selection
        elif self.select_mode == "best":
            selected = torch.argmax(selectables_logits, dim=-1)
        elif self.select_mode == "sample":
            selected = self.sampler(selectables_logits)
        else:
            raise RuntimeError("Selection did not go well")
        applied_selection = self.selector((inputs, selected))
        return applied_selection, selected

    def _targetables(self, inputs):
        predict_targetables = self.predict_targetables_encoders(inputs)
        predict_targetables = self.predict_targetables_linear(predict_targetables)
        predict_targetables = self.predict_targetables_flatten(predict_targetables)
        return predict_targetables


    # Other model methods
    def selectables(self, inputs):
        
        # 1. Embed the piece, player and position information
        embeding = self._embed(inputs)
        
        # 2. Process this information
        shared_processed = self._process(embeding)
        del embeding
        # 3. Predict selectables
        predict_selectables_logits = self._selectables(shared_processed)

        return predict_selectables_logits, shared_processed

    def targetables(self, inputs, selectables_logits=None, selection=None):
        # 4. Selection
        applied_selection, _ = self._select(inputs, 
            selectables_logits=selectables_logits, selection=selection)
        
        # 5. Predict targetables
        predict_targetables_logits = self._targetables(applied_selection)
        
        return predict_targetables_logits

    def weakly_load_state_dict(self, state_dict: dict) -> None:
        model_dict = self.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(filtered_state_dict)
        self.load_state_dict(model_dict)

def preprocess_for_move_polices_and_values(moves, board_shape):
    # Froms
    froms_indices = index_position([move[:,0,:] for move in moves], board_shape)
    # Unique froms
    unique_froms_indices = [np.unique(f_indices) for f_indices in froms_indices]
    # Count
    n_froms = [len(unique_f_indices) for unique_f_indices in unique_froms_indices] 
    # Reduce
    r_froms = [] 
    for unique_f_indices in unique_froms_indices:
        r_froms += list(unique_f_indices)
    # Tos
    tos_indices = index_position([move[:,1,:] for move in moves], board_shape)

    return froms_indices, tos_indices, n_froms, r_froms

def postprocess_for_move_polices_and_values(predict_selectables_probs, predict_targetables_probs, n_froms, r_froms, froms_indices, tos_indices):
    
    # Get probabilies
    policies = []
    sample_to_i = 0
    for sample_from_i, (sample_froms_indices, sample_tos_indices) in enumerate(zip(froms_indices, tos_indices)):
        sp = predict_selectables_probs[sample_from_i].take(sample_froms_indices)
        tp = np.zeros(len(sample_tos_indices))
        for _ in range(n_froms[sample_from_i]):
            from_index = r_froms[sample_to_i]
            is_to = sample_froms_indices == from_index
            tp[is_to] = predict_targetables_probs[sample_to_i].take(sample_tos_indices[is_to])
            sample_to_i+=1
        props = sp * tp
        policies.append(props)  
    return policies

def index_position(positions, board_shape):
    indices = [board_shape[1]*position[:,0]+position[:,1] for position in positions]
    return indices

if __name__ == "__main__":

    board_size = (8,8)
    batch_size = 16

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    chessformer = Chessformer(board_shape=board_size)
    chessformer.to(device)


    pieces = torch.randint(0, 1, (batch_size, *board_size), dtype=torch.int64).to(device)
    players = torch.randint(0, 1, (batch_size, *board_size), dtype=torch.int64).to(device)
    current_player = torch.randint(0, 1, (batch_size, *board_size), dtype=torch.int64).to(device)
    inputs = pieces, players, current_player
    outputs = chessformer(inputs)
    