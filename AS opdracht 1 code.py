import random
from dataclasses import dataclass
import numpy as np


@dataclass
class State:
    x: int
    y: int
    reward: int
    value: int
    next_value: int  # for value iteration
    is_end: bool


class Maze:
    def __init__(self):
        self.finish_locations = [(0, 0), (3, 3)]
        self.grid = {}
        self.grid_size = 4
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if (x, y) == (3, 0):
                    self.grid[x, y] = State(x=x, y=y, reward=40, value=0, next_value=0, is_end=True)
                elif (x, y) == (0, 3):
                    self.grid[x, y] = State(x=x, y=y, reward=10, value=0, next_value=0, is_end=True)
                elif (x, y) == (1, 3):
                    self.grid[x, y] = State(x=x, y=y, reward=-2, value=0, next_value=0, is_end=False)
                elif (x, y) in [(2, 1), (3, 1)]:
                    self.grid[x, y] = State(x=x, y=y, reward=-10, value=0, next_value=0, is_end=False)
                else:
                    self.grid[x, y] = State(x=x, y=y, reward=-1, value=0, next_value=0, is_end=False)  # reward of location

    def step(self, current_state: State, direction: tuple):
        home_coord = (current_state.x, current_state.y)
        other_directions = [(0, -1), (0, 1),
                            (-1, 0), (1, 0)]  # What is added or removed to a coord to simulate a direction
        other_directions.remove(direction)

        if random.random() < 0.3:
            direction = random.choice(other_directions)

        neighbour_coords = np.array(direction) + np.array(home_coord)  # add home coords with 1
        neighbour_coords = tuple(np.clip(neighbour_coords, 0, 3))  # if out of bounds: revert to home coords
        return self.grid[neighbour_coords]

    def __str__(self):
        return_string = "\033[93m[Value]\033[35m per square: \n"
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                return_string += "|\033[93m {:<4} \033[35m|".format(round(self.grid[(x, y)].value, 1))
            return_string += "\n"

        return return_string


class Agent:
    def __init__(self):
        self.maze = Maze()
        self.current_state = self.maze.grid[(1, 3)]  # starts at (1, 3)
        self.directions = [(0, -1), (0, 1),
                           (-1, 0), (1, 0)]  # What is added or removed to a coord to simulate a direction
        self.discount = 1
        self.policy = Policy()
        self.is_done = self.current_state.is_end

    def value_iteration(self):
        delta = float("inf")
        k = 0
        print("k = {}".format(k))
        print(self.maze)
        while delta > 0.1:
            delta = 0
            for i in self.maze.grid:
                if not self.maze.grid[i].is_end:  # if it isn't an end state
                    value = self.maze.grid[i].value
                    value_func_neighbours = []

                    # calculate value of state
                    for direction in self.directions:
                        wrong_directions = list(self.directions)
                        wrong_directions.remove(direction)
                        neighbour_state = self.get_neigbour_state(direction, i)  # get state of neighbouring squares
                        value_func = 0.7 * (neighbour_state.reward + (self.discount * neighbour_state.value))
                        for wrong_direction in wrong_directions:
                            neighbour_state = self.get_neigbour_state(wrong_direction, i)  # get state of neighbouring squares
                            value_func += 0.1 * (neighbour_state.reward + (self.discount * neighbour_state.value))
                        value_func_neighbours.append(value_func)

                    self.maze.grid[i].next_value = max(value_func_neighbours)  # set the value in next_value

                    # set new delta
                    if abs(self.maze.grid[i].next_value - value) > delta:
                        delta = abs(self.maze.grid[i].next_value - value)

            # update all values
            for i in self.maze.grid:
                self.maze.grid[i].value = self.maze.grid[i].next_value

            # print new grid values
            k += 1
            print("k = {}".format(k))
            print(self.maze)

    def get_neigbour_state(self, coord_addition: tuple, home_coord: tuple) -> dataclass:
        neighbour_coords = np.array(coord_addition) + np.array(home_coord)  # add home coords with 1
        neighbour_coords = tuple(np.clip(neighbour_coords, 0, 3))  # if out of bounds: revert to home coords
        return self.maze.grid[neighbour_coords]

    def do_action(self):
        best_directions = self.policy.select_action(self.maze.grid, self.current_state)
        best_direction = random.choice(best_directions)
        self.current_state = self.maze.step(self.current_state, best_direction)
        self.is_done = self.current_state.is_end


class Policy:
    def __init__(self):
        self.directions = [(0, -1), (0, 1),
                           (-1, 0), (1, 0)]  # What is added or removed to a coord to simulate a direction
        self.discount = 1

    def select_action(self, grid: dict, current_state: State) -> list:
        best_directions = self.directions
        home_coords = (current_state.x, current_state.y)
        values = []

        # calculate best direction
        for direction in self.directions:
            # Get value of each neighbour
            neighbour_coords = np.array(direction) + np.array(home_coords)  # add home coords with 1
            neighbour_coords = tuple(np.clip(neighbour_coords, 0, 3))  # if out of bounds: revert to home coords
            neighbour_state = grid[neighbour_coords]
            value_func = neighbour_state.reward + (self.discount * neighbour_state.value)
            values.append(value_func)

        max_value = max(values)
        indices = [index for index, value in enumerate(values) if value == max_value]
        best_directions = list(map(best_directions.__getitem__, indices))  # get all best directions

        return best_directions

    def print_policy(self, grid: dict):
        direction_arrows = ["↑", "↓", "←", "→"]
        print_str = "Policy: \n"
        for y in range(4):
            for x in range(4):
                if grid[(x, y)].is_end is False:
                    directions = self.select_action(grid, grid[(x, y)])
                    indices = [index for index, value in enumerate(self.directions) if value in directions]
                    arrows = "".join(list(map(direction_arrows.__getitem__, indices)))  # get all best directions
                    print_str += "|\033[93m {:<4} \033[35m|".format(arrows)
                else:
                    print_str += "|\033[93m {:<4} \033[35m|".format("X")
            print_str += "\n"
        print(print_str)


if __name__ == "__main__":
    agent = Agent()

    print("\033[35mValue iteration start:")
    agent.value_iteration()
    print("Value iteration complete.")

    agent.policy.print_policy(agent.maze.grid)

    print("Starting sample run. Start location:\033[93m {}".format((agent.current_state.x, agent.current_state.y)))
    while agent.is_done is False:
        print("\033[93m{}".format((agent.current_state.x, agent.current_state.y)), end="\033[35m -> ")
        agent.do_action()
    print("\033[93m{}".format((agent.current_state.x, agent.current_state.y)), end="\033[35m, ")
    print("End state reached")
