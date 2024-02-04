import torch
from chess_royale_ai import Chessformer, ChessRoyaleGamesProcess
from chess_royale_environment import Game
from chess_royale_environment.boards import RandomBoard
from torch.multiprocessing import Manager
import time
import numpy as np
import gc

from chess_royale_ai.transformer_network import postprocess_for_move_polices_and_values, preprocess_for_move_polices_and_values

# sudo fuser -v /dev/nvidia*

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    CUDA_LAUNCH_BLOCKING=1
    board_size = (9,9)
    number_of_games = 128
    batch_size = 32
    results_batch_size = 32
    hill_hold_duration = 2
    n_descent = 400
    max_samples_per_game = 4
    number_of_processes = 3
    # load_path = "./weights/self_supervised"
    load_path = "./weights/self_play"
    save_path = "./weights/self_play"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    chessformer = Chessformer(board_shape=board_size)
    # chessformer = Chessformer(board_shape=board_size, embed_dim=16, num_heads=4, shared_encoder_layers=2)
    chessformer.load_state_dict(torch.load(load_path))
    chessformer.to(device)

    manager = Manager()
    values_lookup = manager.dict()
    policies_lookup = manager.dict()

    processes = [ChessRoyaleGamesProcess(
        device,
        number_of_games=number_of_games, 
        n_descent=n_descent, 
        batch_size=batch_size,
        results_batch_size=results_batch_size,
        hill_hold_duration=hill_hold_duration,
        max_samples_per_game=max_samples_per_game,
        always_last_sample=True,
        values_lookup=values_lookup,
        policies_lookup=policies_lookup,) for _ in range(number_of_processes)]
    for process in processes:
        time.sleep(5)
        process.start()

    select_loss = torch.nn.CrossEntropyLoss()
    target_loss = torch.nn.CrossEntropyLoss()
    value_loss = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(chessformer.parameters(), lr=1e-5)

    running_loss = 0.0
    train_counter: int = 0
    update_period:int = 10
    current_time = time.time()
    start_time = current_time
    execute_time = 0
    while True:
        for process in processes:
            if process.has_request():
                # current_time = time.time()
                # process.grant_resource_access()
                # start_time = time.time()
                # print(time.time() - start_time)
                
                # Put shizzle to the GPU
                # inputs = [model_input.to(device) for model_input in inputs]
                # selection_filters = selection_filters.to(device)
                # r_froms_tensor = r_froms_tensor.to(device)
                # n_froms_tensor = n_froms_tensor.to(device)
                with torch.no_grad():
                    inputs, selection_filters, r_froms_tensor, n_froms_tensor = process.get_request()

                    # Get policies and logits
                    predict_selectables_logits, predict_targetables_logits, values = chessformer._move_policies_and_values(inputs, n_froms_tensor, r_froms_tensor)
                    
                    batch_size = predict_selectables_logits.shape[0]

                    # Apply selection filter
                    if selection_filters is not None:
                        predict_selectables_logits[selection_filters.view(batch_size, -1)==0] = 0
                    predict_selectables_probs = torch.softmax(predict_selectables_logits, dim=-1)
                    predict_targetables_probs = torch.softmax(predict_targetables_logits, dim=-1)

                    process.put_response((
                        predict_selectables_probs.cpu().numpy(), 
                        predict_targetables_probs.cpu().numpy(),
                        values.cpu().numpy())
                    )
                # gc.collect()
                # Monitoring
                # new_time = time.time()
                # delta_time = new_time - current_time
                # execute_time += delta_time
                # if np.random.randint(10) == 0:
                #     # print(delta_time)
                #     print(execute_time/(start_time - current_time))
                # current_time = new_time
                # process.put_response((policies, values.cpu().numpy()))
            if process.has_results():
                # process.grant_resource_access()
                # start_time = time.time()
                inputs, selection_policy_matrices, target_policy_matrices, froms, zs = process.get_results()
                # print(time.time() - start_time)
                batch_size = len(froms)
                # Train
                optimizer.zero_grad()
                # Predict
                outputs = chessformer(inputs, selection=froms)
                # Get Lost
                s_loss = select_loss(outputs[0].view(batch_size, -1), selection_policy_matrices.view(batch_size, -1))
                t_loss = target_loss(outputs[1].view(batch_size, -1), target_policy_matrices.view(batch_size, -1))
                v_loss = value_loss(outputs[3], zs)
                loss = s_loss + t_loss + v_loss
                # Update
                loss.backward()
                optimizer.step()
                # Monitor
                running_loss += loss.item()
                current_step_index = train_counter % update_period
                print(current_step_index, running_loss / (current_step_index+1), s_loss.item(), t_loss.item(), v_loss.item())
                if current_step_index == 9:
                    running_loss = 0.0
                torch.save(chessformer.state_dict(), save_path)
                # Update train counter
                train_counter += 1
                # Clear the memmory
                values_lookup.clear()
                policies_lookup.clear()






    # def give_randomly_access():
    #     wants_gpu_access = [process for process in processes if process.wants_resource_request()]
    #     n_wants = len(wants_gpu_access)
    #     if n_wants == 0: return
    #     lucky_process = wants_gpu_access[np.random.choice(range(n_wants))]
    #     lucky_process.grant_resource_access()

    # previous_time = time.time()
    # while True:
    #     time.sleep(0)
    #     # Update how many have access
    #     current_gpu_users = np.sum([process.has_resource_access() for process in processes])
    #     if current_gpu_users < max_number_of_gpu_users:
    #         give_randomly_access()

    #         current_time = time.time()
    #         if current_time - previous_time > 60:
    #             torch.save(chessformer.state_dict(), save_path)
    #             previous_time = current_time
    #             self.ask_resource_access()

    #             self.hold_until_resource_acces()
    #             self.release_resource()
    #             self.handle_request(inputs, selection_filters, r_froms_tensor, n_froms_tensor)

    #             self.release_resource()