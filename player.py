#!/usr/bin/env python3
import math
import time




import random

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return




class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate game tree object
        first_msg = self.receiver()
        # Initialize your minimax model
        model = self.initialize_model(initial_data=first_msg)

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(
                model=model, initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def initialize_model(self, initial_data):
        """
        Initialize your minimax model 
        :param initial_data: Game data for initializing minimax model
        :type initial_data: dict
        :return: Minimax model
        :rtype: object

        Sample initial data:
        { 'fish0': {'score': 11, 'type': 3}, 
          'fish1': {'score': 2, 'type': 1}, 
          ...
          'fish5': {'score': -10, 'type': 4},
          'game_over': False }

        Please note that the number of fishes and their types is not fixed between test cases.
        """
        # EDIT THIS METHOD TO RETURN A MINIMAX MODEL ###
        return None

    def search_best_next_move(self, model, initial_tree_node):
        """
        Use your minimax model to find best possible next move for player 0 (green boat)
        :param model: Minimax model
        :type model: object
        :param initial_tree_node: Initial game tree node 
        :type initial_tree_node: game_tree.Node 
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE FROM MINIMAX MODEL ###
        
        # NOTE: Don't forget to initialize the children of the current node 
        #       with its compute_and_get_children() method!

        node = initial_tree_node
        node.children = node.compute_and_get_children()
        result = []
        max_util = -100000
        action = 0
        start_time = time.time()
        for child in node.children:
            result = alpha_beta_pruning(child, start_time, max_depth=4)
            if result[0]>max_util:
                max_util = result[0]
                action = result[1]
        if max_util == 0:
            return ACTION_TO_STR[random.randrange(5)]
        return ACTION_TO_STR[action]

def alpha_beta_pruning(child, start_time, max_depth):
    value= max_value(child, max_depth, -math.inf, math.inf, start_time)
    return value, child.move

def max_value(node, max_depth, alpha, beta, start_time):
    if time.time() - start_time> 0.05:
        return evaluate(node)

    node.children = node.compute_and_get_children()
    if node.depth == max_depth:
        return evaluate(node)

    sorted_children = node.children
    sorted_children.sort(key=lambda x: evaluate(x), reverse=True)
    value = -math.inf
    for child in sorted_children:
        value = max(value, min_value(child, max_depth, alpha, beta, start_time))

        if value >= beta:
            return value
        alpha = max(alpha,value)
    return value

def min_value(node, max_depth, alpha, beta, start_time):
    if time.time() - start_time> 0.05:
        return evaluate(node)

    node.children = node.compute_and_get_children()
    if node.depth == max_depth:
        return evaluate(node)

    sorted_children = node.children
    sorted_children.sort(key=lambda x: evaluate(x), reverse=True)
    value = math.inf
    for child in sorted_children:
        value = min(value, max_value(child, max_depth, alpha, beta, start_time))
        if value <= alpha:
            return value
        beta = min(beta,value)
    return value

def evaluate(node):
    state = node.state
    in_game_fish = [[0 for i in range(2)] for j in range(len(state.get_fish_positions()))]
    for j,key in enumerate(state.get_fish_positions()):
        in_game_fish[j][0] = state.get_fish_scores()[key]
        in_game_fish[j][1] = key
    if len(in_game_fish)==0:
        return 0
    hookPos = state.get_hook_positions()[0]
    oppPos = state.get_hook_positions()[1]
    distance = [0.0 for i in range(len(in_game_fish))]
    reachable = [0.0 for i in range(len(in_game_fish))]
    scores = [0.0 for i in range(len(in_game_fish))]
    for i in range(len(in_game_fish)):
        fishPos = state.get_fish_positions()[in_game_fish[i][1]]
        oppDistance = manhattanDistance(oppPos, fishPos)
        distance[i] = manhattanDistance(hookPos, fishPos)
        if distance[i]>oppDistance:
            reachable[i]= 2*distance[i]
        elif distance[i]==0:
            reachable[i] = 0.01
        else:
            reachable[i]= distance[i]

        scores[i]= in_game_fish[i][0]/reachable[i]

    scorePlayerMax= node.state.get_player_scores()[0]
    scorePlayerMin= node.state.get_player_scores()[1]
    myCaught = state.get_caught()[0]
    oppCaught = state.get_caught()[1]
    if myCaught is None:
        myScoreCaught = 0
    else:
        myScoreCaught = state.get_fish_scores()[myCaught]
    if oppCaught is None:
        oppScoreCaught= 0
    else:
        oppScoreCaught= state.get_fish_scores()[oppCaught]
    heuristic = sum(scores)/len(scores) + scorePlayerMax + 10*myScoreCaught - scorePlayerMin - 10*oppScoreCaught
    return heuristic


def manhattanDistance(hookPos, fishPos):
    return abs(hookPos[0]- fishPos[0])+abs(hookPos[1]- fishPos[1])

