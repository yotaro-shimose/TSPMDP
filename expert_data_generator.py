
import pickle
from copy import deepcopy

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def create_transition(manager, routing, solution, nodes, factor):
    index = routing.Start(0)
    # initial process
    depo = manager.IndexToNode(index)
    current_mask = np.ones((nodes.shape[0],), dtype=np.int)
    current_mask[depo] = 0
    data = []
    while not routing.IsEnd(index):
        step_dict = {}
        step_dict["mask"] = deepcopy(current_mask)
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        step_dict["action"] = np.array(
            [manager.IndexToNode(index)], dtype=np.int)
        current_mask[manager.IndexToNode(index)] = 0
        step_dict["reward"] = np.array(
            [-routing.GetArcCostForVehicle(previous_index, index, 0)]) / factor
        step_dict["status"] = np.array(
            [manager.IndexToNode(previous_index), depo], dtype=np.int)
        step_dict["graph"] = nodes
        step_dict["next_status"] = np.array(
            [manager.IndexToNode(index), depo], dtype=np.int)
        step_dict["next_mask"] = deepcopy(current_mask)
        step_dict["done"] = np.array([0], dtype=np.int)
        data.append(step_dict)

    final_step = data.pop(-1)

    final_reward = final_step["reward"]
    data[-1]["reward"] += final_reward
    data[-1]["done"] = 1

    return data


def create_instance(n_nodes, factor):
    # N
    nodes = np.random.random((n_nodes, 2))
    distance = np.sqrt(
        ((nodes[np.newaxis, :, ...] - nodes[:, np.newaxis, ...])**2).sum(axis=-1))
    integer_distance = np.floor(distance * factor).astype(np.int)
    data = {}
    data['distance_matrix'] = integer_distance
    data['num_vehicles'] = 1
    data['depot'] = 0
    return nodes, data


def create_expert_episode(n_nodes, factor=10000):
    n_nodes = 100
    factor = 10000
    raw_nodes, data = create_instance(n_nodes, factor)
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # get solution vector
    return create_transition(
        manager, routing, solution, raw_nodes, factor)


def create_expert_data(n_samples, n_nodes, factor=10000):

    data = []
    for _ in range(n_samples):
        data += create_expert_episode(n_nodes)

    return data


def save_expert_data(file_path, data):
    with open(file_path, mode='wb') as f:
        pickle.dump(data, f, protocol=4)


def load_expert_data(file_path):
    loaded_data = None
    with open(file_path, mode='rb') as f:
        loaded_data = pickle.load(f)
    if loaded_data:
        return loaded_data
    else:
        raise ValueError("data is None")
