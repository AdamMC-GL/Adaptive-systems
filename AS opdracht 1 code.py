import random
from dataclasses import dataclass
import numpy as np


@dataclass
class State:
    """A state, used in the grid of the Maze class to store all information of each square."""
    x: int
    y: int
    reward: int
    value: int
    next_value: int  # for value iteration
    is_end: bool


class Maze:
    """The maze, the environment of the agent. Contains a grid with states and coordinates, and a function that lets
    the agent do a step within the maze."""
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

    def step(self, current_state: State, direction: tuple) -> State:
        """
            Lets the agent do a step in the maze given a current state and direction an returning a
            state back. Each step taken has a possibility of not succeeding 30% of the time. In this case the
            function will return a different next state.
        Parameters
        ----------
            current_state (State dataclass): The current state of the agent.
            direction (tuple): The direction the agent want to try to go to
        Returns
        -------
             A state for the agent to become its new state as. Can be the wanted state, or
             one of the 3 other possible neighbour states.
        """
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
        """Prints out the entire grid and the values of each square."""

        return_string = "\033[93m[Value]\033[35m per square: \n"
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                return_string += "|\033[93m {:<4} \033[35m|".format(round(self.grid[(x, y)].value, 1))
            return_string += "\n"

        return return_string


class Agent:
    """
    The agent class, contains the maze, its current state, a list of possible directions it can go,
    the discount used for the value function, an instance of the policy, and a boolean that keeps track
    if its on an end_state.

    Has a function to do value iteration on the maze, a function to get the neighbours of a given state,
    and a function that does an action for the agent using its policy.
    """
    def __init__(self):
        self.maze = Maze()
        self.current_state = self.maze.grid[(1, 3)]  # starts at (1, 3)
        self.directions = [(0, -1), (0, 1),
                           (-1, 0), (1, 0)]  # What is added or removed to a coord to simulate a direction
        self.discount = 1
        self.policy = Policy()
        self.is_done = self.current_state.is_end

    def value_iteration(self):
        """
        Value iteration of the model. For each state in the grid it calculates the
        value by taking each neighbours reward and value, and using them to for the
        value function. It has a probability of its chosen next state to succeed with
        0.7 chance. Each other probability has an equal 0.1 chance each.

        So for each possible direction it uses the following value function:
        0.7 * (wanted next state + discount * value next state) +
        0.1 * (unwanted next state + discount * value next state) +
        0.1 * (unwanted next state + discount * value next state) +
        0.1 * (unwanted next state + discount * value next state) +

        Repeat for each direction.
        """
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

    def get_neigbour_state(self, coord_addition: tuple, home_coord: tuple) -> State:
        """
            Gives the neighbour state of a given state. Gets a tuple to add to its home coordinates, and clips it
            back to >0 and <4 to make sure the coordinates cannot go out of bounds.
        Parameters
        ----------
            coord_addition (tuple): A set of numbers that is added to the home_coord
            home_coord (tuple): The coordinates of a state the agent wants to know the neighbours from.
        Returns
        -------
             One state next to the state on home_coord
        """
        neighbour_coords = np.array(coord_addition) + np.array(home_coord)  # add home coords with 1
        neighbour_coords = tuple(np.clip(neighbour_coords, 0, 3))  # if out of bounds: revert to home coords
        return self.maze.grid[neighbour_coords]

    def do_action(self):
        """
            Lets the agent do an action within the grid. Decides the best direction with
            the policy, then uses that direction to make a step within the maze. Updates on
            each step if the agent is on the end state or not.
        """
        best_directions = self.policy.select_action(self.maze.grid, self.current_state)
        best_direction = random.choice(best_directions)  # From all equally best options it chooses one randomly
        self.current_state = self.maze.step(self.current_state, best_direction)
        self.is_done = self.current_state.is_end


class Policy:
    """The policy of the agent."""
    def __init__(self):
        self.directions = [(0, -1), (0, 1),
                           (-1, 0), (1, 0)]  # What is added or removed to a coord to simulate a direction
        self.discount = 1

    def select_action(self, grid: dict, current_state: State) -> list:
        """
            Selects the best actions to take from a given state, looks at each neighbour and
            uses the value function to determine which neighbouring state are the best,
            If more than one action is the best, all of the best actions will be
            returned.
        Parameters
        ----------
            grid (dictionary): The grid of the Maze class containing all the states
            current_state (State dataclass): The grid of the Maze class containing all the states
        Returns
        -------
             A list of all best direction to go
        """
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
        """
        Prints out the entire grid and its policy, with each square showing all the best directions
        to go to from that square.
        Parameters
        ----------
            gird (dictionary): The grid of the Maze class containing all the states
        """
        direction_arrows = ["↑", "↓", "←", "→"]
        print_str = "\033[93mPolicy: \033[35m\n"
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
