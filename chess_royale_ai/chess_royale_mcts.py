from typing import Tuple
import time
import numpy as np
from matplotlib import pyplot as plt
from mcts.processes import MultiplayerMCTSSessionsProcess
from mcts.nodes import PolicyValueNode
from mcts.trees import PolicyValueTree, visualize_mcts_tree
from chess_royale_environment.games import Game
from chess_royale_environment.boards import Board, RandomBoard
from chess_royale_environment.players import Player
from chess_royale_environment.win_conditions import KingOfTheHill, LastKingStanding

import torch

from chess_royale_ai.transformer_network import postprocess_for_move_polices_and_values, preprocess_for_move_polices_and_values


class GamePolicyValueTree(PolicyValueTree):

    def add_to_lookup_tables(self, node):
        # print(node.string_notation)
        string_notation = " ".join(node.string_notation.split(" ")[:-1])
        # print(string_notation)
        # string_notation = node.string_notation
        self.values[string_notation] = node.values
        self.policies[string_notation] = node.policy

class ChessRoyaleGamesProcess(MultiplayerMCTSSessionsProcess):

    def __init__(self, device,
        number_of_games:int=16, board_size=(9,9), 
        hill_hold_duration:int=2,
        max_samples_per_game=5, always_last_sample=True,
        **kwargs):

        super(ChessRoyaleGamesProcess, self).__init__(**kwargs)

        self.board_size: Tuple[int, int] = board_size
        self.number_of_games: int = number_of_games

        self.games = {}
        self.history = {}
        self.hill_hold_duration:int = hill_hold_duration

        self.max_samples_per_game: int = max_samples_per_game
        self.always_last_sample: bool = always_last_sample

        self.win_conditions = [KingOfTheHill(number_of_rounds=self.hill_hold_duration), LastKingStanding()]
        self.device = device
        
    def run(self):
        super(ChessRoyaleGamesProcess, self).run()
        
        while len(self.trees) < self.number_of_games:
            self.add_new_game()

        # For monitoring
        import time
        times = np.zeros(5, dtype=float)
        current_time = time.time()
        last_time = current_time
        counter = 0
        num_descents = 0.0
        num_total = 0.0

        while self.running:
            # print(
            #     np.sum([tree.descending for tree in self.trees]),
            #     np.sum([tree.is_finished for tree in self.trees]),
            #     len(self.trees),
            #     len(self.finished_trees)
            # )
            # active_trees = np.sum([tree.descending for tree in self.trees])
            # if active_trees == 0:
            #     time.sleep(0)
            # # Monitoring
            # num_descents += active_trees
            # num_total += len(self.trees)
            # current_time = time.time()


            # Perform a step in all trees
            self.step()

            # # Monitoring
            # next_time = time.time()
            # times[0] += next_time - current_time
            # current_time = next_time
            
            # Handle leave nodes
            leave_trees = self.get_leave_trees()
            self.handle_leave_nodes(leave_trees)

            # # Monitoring
            # next_time = time.time()
            # times[1] += next_time - current_time
            # current_time = next_time

            # Get batch for evaluation
            request_trees_batch = self.get_request_batch()
            if len(request_trees_batch) > 0:
                self.handle_request_batches(request_trees_batch)

            # # Monitoring
            # next_time = time.time()
            # times[2] += next_time - current_time
            # current_time = next_time

            # Handle response
            if self.has_response():
                # Get the response from the neural network process
                response, request, trees = self.get_response()
                self.handle_responses(response, request, trees)

            # # Monitoring
            # next_time = time.time()
            # times[3] += next_time - current_time
            # current_time = next_time

            # Handle finished tree searches
            finished_trees = self.get_finished_trees()
            if len(finished_trees) > 0:
                self.handle_finished_games(finished_trees)

            # # Monitoring
            # next_time = time.time()
            # times[4] += next_time - current_time
            # current_time = next_time

            # Handle when we have enough trainings samples
            results_batch = self.get_results_batch()
            # Only continue if there are results
            if len(results_batch) != 0:
                self.handle_result_batches(results_batch)

            # # Monitoring
            # next_time = time.time()
            # times[4] += next_time - current_time
            # current_time = next_time

            # # Monitoring results
            # if counter % 100_000 == 0:
            #     print(np.round(100 * times/np.sum(times)), current_time - last_time, num_descents/num_total)
            #     last_time = current_time
            # counter += 1
            

    def handle_leave_nodes(self, leave_trees):
        for leave_tree in leave_trees:
            # Check if we are dealing with a new tree
            if leave_tree.current_node is None:
                game: Game = self.games[leave_tree]
                if game.is_finished:
                    raise RuntimeError("Root node is also a termination node")
            else:
                game: Game = Game.from_string_notation(leave_tree.current_node.string_notation,
                    win_conditions=self.win_conditions)
                # Get the action index proposed by the mcts
                action_index = leave_tree.current_action_index
                # Get the action
                action = leave_tree.current_node.actions[action_index]
                f, t = action
                # Take the action
                game.step(f, t)
                # Handle the case that we ended up in a termination node
                if game.is_finished:
                    # Create a new node for the new state
                    string_notation = game.string_notation()#" ".join(game.string_notation().split(' ')[:-1])
                    # Get the values belongint to the termination state
                    if len(game.winners) > 0:
                        values = -np.ones(self.number_of_players)
                        for winner in game.winners:
                            values[winner.value - 1] = 1
                    else:
                        values = np.zeros(self.number_of_players)
                    # Add a new node to the tree
                    node = PolicyValueNode(
                        values=values, 
                        policy=[], 
                        actions=[], 
                        player=game.current_player.value-1, 
                        num_of_players=self.number_of_players, 
                        termination=game.is_finished,
                        string_notation=string_notation, 
                        annotation=str(game))
                    leave_tree.add_node(action_index, node)
                    continue
            string_notation = game.string_notation()
            policy, values = self.get_policy_and_value(string_notation)
            if policy is not None and values is not None:
                actions = game.get_legal_moves()
                
                if len(policy) == len(actions) and len(values) == 2:
                    # Get the action index proposed by the mcts
                    action_index = leave_tree.current_action_index
                    # Add a new node to the tree
                    node = PolicyValueNode(
                        values=values, 
                        policy=policy, 
                        actions=actions, 
                        player=game.current_player.value-1, 
                        num_of_players=self.number_of_players, 
                        termination=game.is_finished,
                        string_notation=string_notation, 
                        annotation=str(game))
                    leave_tree.add_node(action_index, node)
                    continue
                else:
                    print(f"denied {time.time()} {len(policy)} {len(actions)}")
            state = game.state()
            moves = game.get_legal_moves()
            leave_tree.storage["state"] = state
            leave_tree.storage["moves"] = moves
            leave_tree.storage["game"] = game
            self.queue_request(leave_tree)

    def get_policy_and_value(self, string_notation):
        string_notation = " ".join(string_notation.split(" ")[:-1])
        if string_notation in self.values_lookup:
            try:
                return self.policies_lookup[string_notation], self.values_lookup[string_notation]
            except:
                return None, None
        return None, None

    def handle_request_batches(self, request_trees_batch):
        number_of_samples = len(request_trees_batch)
        leave_trees, states, batch_moves = [], [], []
        for leave_tree in request_trees_batch:
            leave_trees.append(leave_tree)
            states.append(leave_tree.storage["state"])
            batch_moves.append(leave_tree.storage["moves"])
        # Get the values of the new game states
        inputs = game_states_to_inputs(states)
        # Create the selection filters for the game states
        selection_filters = torch.zeros((number_of_samples, *self.board_size),dtype=torch.int16)
        for sample_index, sample_moves in enumerate(batch_moves):
            for move in sample_moves:
                selection_filters[sample_index, move[0,0], move[0,1]]=1
        # Place request
        # self.put_request((inputs, batch_moves, selection_filters))
        self.froms_indices, self.tos_indices, self.n_froms, self.r_froms = preprocess_for_move_polices_and_values(batch_moves, self.board_size)
        
        r_froms_tensor = torch.tensor(self.r_froms, dtype=torch.int64)
        n_froms_tensor = torch.tensor(self.n_froms, dtype=torch.int64)
        
        # self.ask_resource_access()
        # self.hold_until_resource_access()
        # Put shizzle on the GPU
        with torch.no_grad():
            inputs = [model_input.to(self.device) for model_input in inputs]
            selection_filters = selection_filters.to(self.device)
            r_froms_tensor = r_froms_tensor.to(self.device)
            n_froms_tensor = n_froms_tensor.to(self.device)
            # Make it shared memory
            # inputs = [model_input.share_memory_() for model_input in inputs]
            # selection_filters = selection_filters.share_memory_()
            # r_froms_tensor = r_froms_tensor.share_memory_()
            # n_froms_tensor = n_froms_tensor.share_memory_()
            # self.put_request([1])
            self.put_request((inputs, selection_filters, r_froms_tensor, n_froms_tensor))
            del inputs, selection_filters, r_froms_tensor, n_froms_tensor
        # self.release_resource()

    def handle_responses(self, response, request, trees):
        # Extract the policies and values of the for the games
        # policies, values = response
        predict_selectables_probs, predict_targetables_probs, values = response
        # The original request info
        # inputs, selection_filters, r_froms_tensor, n_froms_tensor = request

        # Post-processing 
        policies = postprocess_for_move_polices_and_values(
            predict_selectables_probs, 
            predict_targetables_probs, 
            self.n_froms, 
            self.r_froms, 
            self.froms_indices, 
            self.tos_indices
        )

        # Let's handle the results individually
        for tree, sample_policy, sample_values in zip(trees, policies, values):
            sample_moves = tree.storage["moves"]
            # Get the action index proposed by the mcts
            action_index = tree.current_action_index
            # The current game environment
            game: Game = tree.storage["game"]
            if game.is_finished:
                # When dealing with a termination state
                print("this shouldn't happen")
            # Create a new node for the new state
            string_notation = game.string_notation() #" ".join(game.string_notation().split(' ')[:-1])
            # Add a new node to the tree
            node = PolicyValueNode(
                values=sample_values, 
                policy=sample_policy, 
                actions=sample_moves, 
                player=game.current_player.value-1, 
                num_of_players=self.number_of_players, 
                termination=game.is_finished,
                string_notation=string_notation, 
                annotation=str(game))
            tree.add_node(action_index, node)

    def handle_finished_games(self, finished_trees):
        for finished_tree in finished_trees:
            # Get the game
            game: Game = self.games[finished_tree]
            # remove the link of the old tree with the game
            del self.games[finished_tree]
            # Get some nice values
            state = game.state()
            policy = finished_tree.accumulated_policy()
            moves = finished_tree.root_node.actions
            action_index = np.argmax(policy)
            action = moves[action_index]
            # Perform action
            game.step(action[0], action[1])
            # Check the aftermath
            z = -torch.ones(self.number_of_players, dtype=torch.float32)
            for winner in game.winners:
                z[winner.value-1] = 1
            # Get useful shizzle
            selection_policy_matrix, target_policy_matrix = self.policy_and_moves_to_select_target_policy_matrices(policy, moves, action)
            # Add stuff to history
            self.history[game].append((state, selection_policy_matrix, target_policy_matrix, action, z)) 
            # self.history[game].append((state, policy, moves, action, z)) 
            # Check if the game is not yet over
            if not game.is_finished:
                new_tree = GamePolicyValueTree(
                    n_descent=self.n_descent, 
                    shared_policies=self.policies_lookup, 
                    shared_values=self.values_lookup
                )
                self.games[new_tree] = game
                self.add_tree(new_tree)
                continue
            # Handle finished games
            training_history = []
            # Add the last step
            if self.always_last_sample:
                training_history.append(self.history[game][-1])
            # A random few other samples
            num_random_samples = np.min((len(self.history[game]), self.max_samples_per_game)) - int(self.always_last_sample)
            training_sample_indices = np.random.choice(
                range(len(self.history[game])-int(self.always_last_sample)), 
                size=num_random_samples, 
                replace=False)
            for training_sample_index in training_sample_indices:
                training_history.append(self.history[game][training_sample_index])
            # Delete game from history
            self.history[game]
            # add a new game
            self.add_new_game()
            for training_sample in training_history:
                self.queue_result(training_sample)

    def handle_result_batches(self, results_batch):
        batch_size = len(results_batch)
        # # Process results
        states = []
        zs = torch.zeros((batch_size, self.number_of_players), dtype=torch.float32)
        selection_policy_matrices = torch.zeros((batch_size, *self.board_size), dtype=torch.float32)
        target_policy_matrices = torch.zeros((batch_size, *self.board_size), dtype=torch.float32)
        froms = torch.zeros((batch_size, 2), dtype=torch.int64)
        for index, (state, selection_policy_matrix, target_policy_matrix, action, z) in enumerate(results_batch):
            selection_policy_matrices[index] = selection_policy_matrix
            target_policy_matrices[index] = target_policy_matrix
            froms[index] = torch.tensor(action[0], dtype=torch.int64)
            states.append(state)
            zs[index] = z
        inputs = game_states_to_inputs(states, hill_hold_value=self.hill_hold_duration)
        # self.ask_resource_access()
        # self.hold_until_resource_access()
        # Put shizzle to the GPU
        inputs = [i.to(self.device) for i in inputs]
        selection_policy_matrices = selection_policy_matrices.to(self.device)
        target_policy_matrices = target_policy_matrices.to(self.device)
        zs = zs.to(self.device)
        froms = froms.to(self.device)

        # self.release_resource()

        self.put_results((inputs, selection_policy_matrices, target_policy_matrices, froms, zs))

        del inputs, selection_policy_matrices, target_policy_matrices, zs, froms

    def policy_and_moves_to_select_target_policy_matrices(self, policy, moves, action):
        # Create clean matrices
        selection_policy_matrix = torch.zeros(self.board_size, dtype=torch.float32)
        target_policy_matrix = torch.zeros(self.board_size, dtype=torch.float32)
        # Fill the matrices
        for move, probability in zip(moves, policy):
            f, t = move
            selection_policy_matrix[f[0], f[1]]+=probability
            if np.array_equal(f, action[0]):
                target_policy_matrix[t[0], t[1]]+=probability
        target_policy_matrix/=torch.sum(target_policy_matrix)
        # Return the values
        return selection_policy_matrix, target_policy_matrix

    def add_new_game(self):
        game = Game(
            board=RandomBoard(num_players=self.number_of_players, size=self.board_size),
            win_conditions=self.win_conditions)
        game.randomize_pieces()
        tree = GamePolicyValueTree(
            n_descent=self.n_descent,
            shared_policies=self.policies_lookup, 
            shared_values=self.values_lookup
        )
        self.games[tree] = game
        self.history[game] = []
        self.add_tree(tree)


def game_states_to_inputs(game_states, hill_hold_value:int=3):
    """Return processed data for the pytorch chessformer model

    Args:
        game_states (game state): Game state 

    Returns:
        pieces: [description]
        players: [description]
        current_player: [description]
        walkable: [description]
        hill: [description]
        hill_hold: [description]
    """    
    number_of_samples = len(game_states)
    _, w, h = game_states[0][0].shape
    pieces = torch.zeros((number_of_samples, w, h),dtype=torch.int64)
    players = torch.zeros((number_of_samples, w, h),dtype=torch.int64)
    current_player = torch.zeros((number_of_samples, w, h),dtype=torch.int64)
    walkable = torch.zeros((number_of_samples, w, h),dtype=torch.int64)
    hill = torch.zeros((number_of_samples, w, h),dtype=torch.float16)
    hill_hold = torch.zeros((number_of_samples, w, h),dtype=torch.float16)
    for index, (board_state, number_of_players, current_player_value, round) in enumerate(game_states):
        pieces[index] = torch.tensor(board_state[0], dtype=torch.int64)
        players[index] = torch.tensor(board_state[1], dtype=torch.int64)
        current_player[index] = current_player_value
        walkable[index] = torch.tensor(board_state[2], dtype=torch.int64)
        hill[index] = torch.tensor(board_state[3], dtype=torch.float16)
        hill_hold[index] = hill[index] * torch.tensor(((hill_hold_value - board_state[4]) / 5), dtype=torch.float16)
    return pieces, players, current_player, walkable, hill, hill_hold

if __name__ == "__main__":

    process = ChessRoyaleGamesProcess()

    games = []
    for _ in range(16):
        game = Game(board=RandomBoard(num_players=2))
        game.randomize_pieces()
        games.append(game)
    process.put_new_environments([game.state() for game in games])
    process.start()

    while True:
        if process.has_request():
            batch = process.get_request()
            print(batch)
            # process.put_response([i for i in range(len(batch))])