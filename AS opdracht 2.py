import random
from dataclasses import dataclass
import numpy as np
import copy
from collections import namedtuple

"""A script showcasing a MDP (Markov Decision Process), contains a State dataclass,
Maze class, Agent class, and a Policy class. The Agent contains an instance of the Maze
and Policy, and a Maze contains a grid of States. Each state contains a reward, and a 
value starting at 0. These will be used by the agent to perform value iteration on the
Maze."""


@dataclass
class State:
    """A state, used in the grid of the Maze class to store all information of each square."""
    x: int
    y: int
    reward: float
    value: float
    next_value: float  # for value iteration
    is_end: bool


class Maze:
    """The maze, the environment of the agent. Contains a grid with states and coordinates, and a function that lets
    the agent do a step within the maze."""
    def __init__(self, grid_size: int):
        self.grid = {}
        self.grid_size = grid_size
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                self.grid[x, y] = State(x=x, y=y, reward=-1, value=0, next_value=0, is_end=False)

        self.grid[(3, 0)] = State(x=3, y=0, reward=40, value=0, next_value=0, is_end=True)  # changed reward and is_end
        self.grid[(0, 3)] = State(x=0, y=3, reward=10, value=0, next_value=0, is_end=True)  # changed reward and is_end
        self.grid[(1, 3)].reward = -2
        self.grid[(2, 1)].reward = -10
        self.grid[(3, 1)].reward = -10

    def step(self, current_state: State, direction: tuple) -> State:
        """
            Lets the agent do a step in the maze given a current state and direction an returning a
            state back. Each step taken has a possibility of not succeeding 30% of the time. In this case the
            function will return a different next state.
        Parameters
        ----------
            current_state (State dataclass): The current state of the agent.
            direction (tuple): The direction the agent wants to try to go to
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
    def __init__(self, grid_size: int, discount: float, start_location: tuple):
        self.grid_size = grid_size
        self.maze = Maze(self.grid_size)
        self.current_state = self.maze.grid[start_location]  # starts at (1, 3)
        self.directions = [(0, -1), (0, 1),
                           (-1, 0), (1, 0)]  # What is added or removed to a coord to simulate a direction
        self.discount = discount
        self.epsilon = 0.1
        self.policy = Policy(self.grid_size, self.discount)
        self.epsilon_policy = Epsilon_policy(self.grid_size)
        self.is_done = self.current_state.is_end

    def value_iteration(self):
        """
        Value iteration of the model. For each state in the grid it calculates the
        value by taking each neighbours reward and value, and using them to for the
        value function. Due to the environment being a non-deterministic environment
        it has a probability of its chosen next state to succeed with
        0.7 chance. Each other probability has an equal 0.1 chance each.

        So for each possible direction it uses the following value function:
        0.7 * (wanted next state + discount * value next state) +
        0.1 * (unwanted next state + discount * value next state) +
        0.1 * (unwanted next state + discount * value next state) +
        0.1 * (unwanted next state + discount * value next state)

        Repeat for each direction.
        """
        delta = float("inf")  # to enter the while loop
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
                        neighbour_state = self.get_neighbour_state(direction, i)  # get state of neighbouring squares
                        value_func = 0.7 * (neighbour_state.reward + (self.discount * neighbour_state.value))
                        for wrong_direction in wrong_directions:
                            neighbour_state = self.get_neighbour_state(wrong_direction, i)  # get state of neighbouring squares
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

    def tabular(self):
        """Tabular TD for estimating V-policy
        Using greedy policy"""
        step_size = 0.1
        for i in range(10000):
            init_state = self.maze.grid[random.choice(list(self.maze.grid.keys()))]  # random start position
            while not init_state.is_end:
                action = random.choice(self.policy.select_action(self.maze.grid, init_state))
                next_state = self.get_neighbour_state(action, (init_state.x, init_state.y))
                R = next_state.reward
                self.maze.grid[(init_state.x, init_state.y)].value += step_size * (R + self.discount*next_state.value - init_state.value)
                init_state = next_state
        print(self.maze)

    def sarsa(self):
        """Sarsa on policy TD control for estimating Q
        Using soft policy"""
        step_size = 0.1
        for i in range(100000):
            current_state = self.maze.grid[random.choice(list(self.maze.grid.keys()))]  # random start position
            current_action = self.epsilon_policy.select_action_Q(current_state)
            while not current_state.is_end:
                next_state = self.get_neighbour_state(current_action, (current_state.x, current_state.y))
                R = next_state.reward
                next_action = self.epsilon_policy.select_action_Q(next_state)
                self.epsilon_policy.Q[((current_state.x, current_state.y), current_action)] += step_size * (R + self.discount * self.epsilon_policy.Q[((next_state.x, next_state.y), next_action)] - self.epsilon_policy.Q[((current_state.x, current_state.y), current_action)])
                current_state = next_state
                current_action = next_action

    def Q_learning(self):
        """Q-learning off policy TD control for estimating policy
        Using soft policy"""
        step_size = 0.1
        for i in range(50000):
            current_state = self.maze.grid[random.choice(list(self.maze.grid.keys()))]  # random start position

            while not current_state.is_end:
                action = self.epsilon_policy.select_action_Q(current_state)
                next_state = self.get_neighbour_state(action, (current_state.x, current_state.y))
                R = next_state.reward
                max_action = max([self.epsilon_policy.Q[((next_state.x, next_state.y), direction)] for direction in self.directions])
                self.epsilon_policy.Q[((current_state.x, current_state.y), action)] += step_size * (R + self.discount * max_action - self.epsilon_policy.Q[((current_state.x, current_state.y), action)])
                current_state = next_state

    def double_Q(self):
        """Double Q-learning for estimating Q1, Q2
        Using soft policy"""
        step_size = 0.1
        Q1 = copy.deepcopy(self.epsilon_policy.Q)
        Q2 = copy.deepcopy(self.epsilon_policy.Q)

        for i in range(50000):
            current_state = self.maze.grid[random.choice(list(self.maze.grid.keys()))]  # random start position
            while not current_state.is_end:
                action = self.epsilon_policy.select_action_double_Q(current_state, Q1, Q2)
                next_state = self.get_neighbour_state(action, (current_state.x, current_state.y))
                R = next_state.reward
                if random.random() > 0.5:
                    action_values = {direction: Q1[((next_state.x, next_state.y), direction)] for direction in self.directions}
                    best_action = max(action_values, key=action_values.get)
                    Q1[((current_state.x, current_state.y), action)] += step_size * (R + self.discount * Q2[((next_state.x, next_state.y), best_action)] - Q1[((current_state.x, current_state.y), action)])
                else:
                    action_values = {direction: Q2[((next_state.x, next_state.y), direction)] for direction in self.directions}
                    best_action = max(action_values, key=action_values.get)
                    Q2[((current_state.x, current_state.y), action)] += step_size * (R + self.discount * Q1[((next_state.x, next_state.y), best_action)] - Q2[((current_state.x, current_state.y), action)])
                current_state = next_state

        print("Q1")
        self.epsilon_policy.print_Q(Q1)
        print("Q2")
        self.epsilon_policy.print_Q(Q2)

    def mc_control(self):
        """On policy first visit monte carlo control for estimating policy
        Using soft policy"""
        for i in range(20000):
            episode = self.make_episode_mc_control()  # make episode
            G = 0
            for index in range(len(episode)-1, -1, -1):  # start at T - 1 to 0
                current_step = (episode[index].state.x, episode[index].state.y)
                G = self.discount*G + episode[index].reward

                # unless the pair S, A appears  in S0, A0, S1, A2...
                if [episode[index].state, episode[index].action] not in [[move.state, move.action] for move in episode[0: index]]:
                    # append G to returns
                    self.epsilon_policy.returns[(current_step, episode[index].action)].append(G)
                    # Q(s, a) <- average(returns(s, a)
                    self.epsilon_policy.Q[(current_step, episode[index].action)] = sum(self.epsilon_policy.returns[(current_step, episode[index].action)]) / len(self.epsilon_policy.returns[(current_step, episode[index].action)])
                    # A* <- argmax Q(s, a)
                    action_values = {direction: self.epsilon_policy.Q[(current_step, direction)] for direction in self.directions}
                    best_action = max(action_values, key=action_values.get)

                    # Update policy chances with epsilon
                    for a in range(len(self.directions)):
                        if self.directions[a] == best_action:
                            self.epsilon_policy.pi[current_step][a] = 1-self.epsilon_policy.epsilon + (self.epsilon_policy.epsilon/4)
                        else:
                            self.epsilon_policy.pi[current_step][a] = (self.epsilon_policy.epsilon/4)

    def make_episode_mc_control(self):
        """Makes and return an episode, used in monte carlo control. Uses epsilon soft policy.
        An episode begins at any random start position, and ends at an end state"""

        Move = namedtuple("move", "state action reward")
        episode = []
        current_state = self.maze.grid[random.choice(list(self.maze.grid.keys()))]  # random start position
        is_done = current_state.is_end
        while is_done is False:
            direction_index = np.random.choice([0, 1, 2, 3], p=self.epsilon_policy.pi[(current_state.x, current_state.y)])
            direction = self.directions[direction_index]
            next_state = self.get_neighbour_state(direction, (current_state.x, current_state.y))
            episode.append(Move(current_state, direction, next_state.reward))
            current_state = next_state
            is_done = current_state.is_end
        return episode

    def monte_carlo_prediction(self):
        """First visit monte carlo prediction for estimating value"""
        returns = {x: [] for x in self.maze.grid}  # empty the dict
        for i in range(10000):
            episode = self.make_episode_mc_prediction()  # make episode
            G = 0
            for index in range(len(episode)-1, -1, -1):  # start at T - 1 to 0
                current_step = (episode[index].state.x, episode[index].state.y)
                G = self.discount*G + episode[index].reward
                if episode[index].state not in [move.state for move in episode[0: index]]:
                    returns[current_step].append(G)
                    self.maze.grid[current_step].value = sum(returns[current_step]) / len(returns[current_step])
        print(self.maze)

    def make_episode_mc_prediction(self) -> list:
        """Makes and return an episode, used in monte carlo prediction. Uses epsilon soft policy.
        An episode begins at any random start position, and ends at an end state"""

        Move = namedtuple("move", "state action reward")
        episode = []
        current_state = self.maze.grid[random.choice(list(self.maze.grid.keys()))]  # random start position
        is_done = current_state.is_end
        while is_done is False:
            if random.random() > self.epsilon:
                best_direction = random.choice(self.policy.select_action(self.maze.grid, current_state))
            else:
                best_direction = random.choice(self.directions)
            next_state = self.get_neighbour_state(best_direction, (current_state.x, current_state.y))
            episode.append(Move(current_state, best_direction, next_state.reward))
            current_state = next_state
            is_done = current_state.is_end
        return episode

    def get_neighbour_state(self, coord_addition: tuple, home_coord: tuple) -> State:
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
        neighbour_coords = tuple(np.clip(neighbour_coords, 0, self.grid_size - 1))  # if out of bounds: revert to home coords
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
    def __init__(self, grid_size: int, discount: float):
        self.grid_size = grid_size
        self.directions = [(0, -1), (0, 1),
                           (-1, 0), (1, 0)]  # What is added or removed to a coord to simulate a direction
        self.discount = discount

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
            neighbour_coords = tuple(np.clip(neighbour_coords, 0, self.grid_size - 1))  # if out of bounds: revert to home coords
            neighbour_state = grid[neighbour_coords]
            value_func = neighbour_state.reward + (self.discount * neighbour_state.value)
            values.append(round(value_func, 1))

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
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if grid[(x, y)].is_end is False:
                    directions = self.select_action(grid, grid[(x, y)])
                    indices = [index for index, value in enumerate(self.directions) if value in directions]
                    arrows = "".join(list(map(direction_arrows.__getitem__, indices)))  # get all best directions
                    print_str += "|\033[93m {:<4} \033[35m|".format(arrows)
                else:
                    print_str += "|\033[93m {:<4} \033[35m|".format("X")
            print_str += "\n"
        print(print_str)


class Epsilon_policy:
    """The epsilon soft policy of the agent."""
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.directions = [(0, -1), (0, 1),
                           (-1, 0), (1, 0)]  # What is added or removed to a coord to simulate a direction
        self.Q = {}
        self.returns = {}
        self.pi = {}
        self.init_Q_pi_and_returns()
        self.epsilon = 0.1

    def init_Q_pi_and_returns(self):
        """initializes the Q values, the pi-chances with 0 for each state and action, and initializes the
        returns list with an empty list for each state and action."""

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                for dir in self.directions:
                    self.Q[((x, y), dir)] = 0
                    self.returns[((x, y), dir)] = []
                if (x, y) == (0, 3) or (x, y) == (3, 0):  # if end state
                    self.pi[(x, y)] = [0, 0, 0, 0]
                else:
                    chances = [0.025, 0.025, 0.025, 0.925]
                    random.shuffle(chances)
                    self.pi[(x, y)] = chances

    def select_action_Q(self, current_state: State) -> tuple:
        """Select an action based on the Q values and the epsilon"""

        state_location = (current_state.x, current_state.y)
        if random.random() > self.epsilon:
            action_values = {direction: self.Q[(state_location, direction)] for direction in self.directions}
            return max(action_values, key=action_values.get)
        return random.choice(self.directions)

    def select_action_double_Q(self, current_state: State, Q1, Q2):
        """Select an action based on two Q values and the epsilon"""

        state_location = (current_state.x, current_state.y)
        if random.random() > self.epsilon:
            action_values = {direction: Q1[(state_location, direction)] + Q2[(state_location, direction)] for direction in self.directions}
            return max(action_values, key=action_values.get)
        return random.choice(self.directions)

    def select_action_pi(self, current_state: State) -> tuple:
        """Select an action based on the pi chances"""
        state_location = (current_state.x, current_state.y)
        return np.random.choice(self.directions, p=self.pi[state_location])[0]

    def print_pi(self):
        """
            Prints out the entire grid and its pi chances, with each coordinate showing all the pi chance of each
            direction.
        """

        print("\033[93mPi Order: ↑, ↓, ←, → \033[35m")
        print_str = "\033[93mPi Policy chances (epsilon soft policy): \033[35m\n"
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                print_str += "[\033[93m{}\033[35m: {:<28}] ".format((x, y), str(self.pi[(x, y)]))
            print_str += "\n"
        print(print_str)

    def print_Q(self, Q):
        """
            Prints out the entire grid and its Q values, with each coordinate showing all the Q values of each
            direction.
        Parameters
        ----------
            Q (dictionary): The dict containing all the Q values
        """
        print("\033[93mDirection corresponding to each Q value: ↑, ↓, ←, →  \033[35m")
        print_str = "\033[93mQ values: \033[35m\n"
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                print_str += "\033[93m{}\033[35m: ".format(str((x, y)))
                for direction in self.directions:
                    print_str += "{:<4}, ".format(round(Q[((x, y), direction)], 1))
            print_str += "\n"
        print(print_str)


if __name__ == "__main__":
    for discount in [0.9, 1]:
        print("\033[93mDiscount = {}\033[35m".format(discount))
        agent1 = Agent(grid_size=4, discount=discount, start_location=(1, 3))
        agent2 = Agent(grid_size=4, discount=discount, start_location=(1, 3))
        agent3 = Agent(grid_size=4, discount=discount, start_location=(1, 3))
        agent4 = Agent(grid_size=4, discount=discount, start_location=(1, 3))
        agent5 = Agent(grid_size=4, discount=discount, start_location=(1, 3))
        agent6 = Agent(grid_size=4, discount=discount, start_location=(1, 3))

        print("\033[35m[Monte carlo prediction] output:")
        agent1.monte_carlo_prediction()
        agent1.policy.print_policy(agent1.maze.grid)

        print("[Tabular] output:")
        agent2.tabular()
        agent2.policy.print_policy(agent2.maze.grid)

        print("[Monte carlo control] output:")
        agent3.mc_control()
        agent3.epsilon_policy.print_pi()

        print("[Sarsa] output:")
        agent4.sarsa()
        agent4.epsilon_policy.print_Q(agent4.epsilon_policy.Q)

        print("[Q-learning] output:")
        agent5.Q_learning()
        agent5.epsilon_policy.print_Q(agent5.epsilon_policy.Q)

        print("[Double Q-learning] output:")
        agent6.double_Q()

        print("All values, Q-values and pi-chances started with 0. All algorithms are in deterministic environments")
        print("======================\n")
