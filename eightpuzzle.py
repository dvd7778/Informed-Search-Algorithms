import search
import random
import time

# Eight Puzzle Problem class from the aimacode repository, 
# slightly modified to include a value function and a Manhattan Distance hueristic function
class EightPuzzle(search.Problem):
    """ The problem of sliding tiles numbered from 1 to 8 on a 3x3 board, where one of the
    squares is a blank. A state is represented as a tuple of length 9, where  element at
    index i represents the tile number  at index i (0 if it's an empty square) """

    def __init__(self, initial, goal=(0, 1, 2, 3, 4, 5, 6, 7, 8)):
        """ Define goal state and initialize a problem """
        super().__init__(initial, goal)
        # Goal indexes dictionary added to facilitate the Manhattan Distance Heuristic calculation
        self.goal_indexes = {}
        for idx, value in enumerate(self.goal):
            self.goal_indexes[value] = idx 

    def find_blank_square(self, state):
        """Return the index of the blank square in a given state"""

        return state.index(0)

    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment """

        possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        index_blank_square = self.find_blank_square(state)

        if index_blank_square % 3 == 0:
            possible_actions.remove('LEFT')
        if index_blank_square < 3:
            possible_actions.remove('UP')
        if index_blank_square % 3 == 2:
            possible_actions.remove('RIGHT')
        if index_blank_square > 5:
            possible_actions.remove('DOWN')

        return possible_actions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """

        # blank is the index of the blank square
        blank = self.find_blank_square(state)
        new_state = list(state)

        delta = {'UP': -3, 'DOWN': 3, 'LEFT': -1, 'RIGHT': 1}
        neighbor = blank + delta[action]
        new_state[blank], new_state[neighbor] = new_state[neighbor], new_state[blank]

        return tuple(new_state)

    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """

        return state == self.goal

    def check_solvability(self, state):
        """ Checks if the given state is solvable """

        inversion = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if (state[i] > state[j]) and state[i] != 0 and state[j] != 0:
                    inversion += 1

        return inversion % 2 == 0

    # Manhattan Distance heuristic function 
    def h(self, node):
        current_state = node.state
        manhattan_distance = 0
        
        # Calculate the Manhattan distance of each square
        for i in range(9):
            if current_state[i] != 0:  
                current_row, current_col = i // 3, i % 3
                goal_index = self.goal_indexes[current_state[i]]
                goal_row, goal_col = goal_index // 3, goal_index % 3
                manhattan_distance += abs(current_row - goal_row) + abs(current_col - goal_col)
        
        return manhattan_distance
            
    # Function to set the initial state to a random one. Used for Random Restart Hill Climb
    def set_random_initial(self):
        # Generate a list of integers from 0 to 8 and shuffle
        numbers = list(range(9))
        random.shuffle(numbers)
        
        # Set initial state to a random state
        self.initial = tuple(numbers)
        
    # Calculate a states value by comparing the amount of values with the same index as the goal state
    def value(self, state):
        value = 0
        
        # Compare each value's index of the state with the corresponding index in the goal state
        for i in range(9):
            # Increment value when a value in the current state is in the same index as the goal state
            if state[i] == self.goal[i]:  
                value += 1
        return value

  
# Steepest Hill Climbing method from the aimacode repository
def steepest_hill_climbing(problem):
    # while True:              ### This line was used for debugging purposes ###
    current = search.Node(problem.initial)
    
    while True:
        if problem.goal_test(current.state):
            return current.state
                
        neighbors = current.expand(problem)
        if not neighbors:
            break
        neighbor = search.argmax_random_tie(neighbors, key=lambda node: problem.value(node.state))
        
        if problem.value(neighbor.state) <= problem.value(current.state):
            break
        current = neighbor
        
    return current.state

# First Choice Hill Climbing method, partially from the aimacode repository.
# Executes a modified version of the hill climb algorithm that moves to the next state better 
# than the current one, even if it is not the most optimal neighbor state.
def first_choice_hill_climbing(problem):
    # while True:              ### This line was used for debugging purposes ###
    current = search.Node(problem.initial)
    
    while True:
        if problem.goal_test(current.state):
            return current.state
                
        neighbors = current.expand(problem)
        # If there are no neighbors then stop the iteration  
        if not neighbors:
            break
        
        better_neighbor = False
        
        # Search for a neighbor better than the current state
        for neighbor in neighbors:
            if problem.value(neighbor.state) > problem.value(current.state):
                current = neighbor
                better_neighbor = True
                break
                
        # If no neighbor is better than the current state stop the iteration       
        if not better_neighbor:
            break
        
    return current.state

# Random Restart hill climb method, partially from the aimacode repository.
# Takes in an amount of restarts as a parameter. The method executes the hill climb algorithm
# from the aimacode repository, but it restarts from a random position each time it doesn't reach
# the goal, while keeping track of the previous state closest to the goal.
# Returns the best encountered state across all restarts.
def random_restart_hill_climbing(problem, restarts=1000):
    # Keeps track of the best encountered state across the restarts
    best_state = problem.initial
    
    for _ in range(restarts):
    # while True:              ### This line was used for debugging purposes ###
        current = search.Node(problem.initial)
        
        while True:
            if problem.value(current.state) > problem.value(best_state):
                best_state = current.state
                if problem.goal_test(best_state):
                    return best_state
                    
            neighbors = current.expand(problem)
            if not neighbors:
                break
            neighbor = search.argmax_random_tie(neighbors, key=lambda node: problem.value(node.state))
            
            if problem.value(neighbor.state) <= problem.value(current.state):
                break
            current = neighbor
        
        problem.set_random_initial()    
    return best_state

# Schedule function from the aimacode repository.
# Modified to have a lam of 0.001 and limit of 5000.
def exp_schedule(k=20, lam=0.001, limit=5000):
    """One possible schedule function for simulated annealing"""
    return lambda t: (k * search.np.exp(-lam * t)) if t < limit else 0

# Simulated Annealing algorithm from the aimacode repository.
# Modified to use the heuristic function (Manhattan Distance) instead of value function.
def simulated_annealing(problem, schedule=exp_schedule()):
    """[Figure 4.5] CAUTION: This differs from the pseudocode as it
    returns a state instead of a Node."""
    current = search.Node(problem.initial)
    for t in range(search.sys.maxsize):
        T = schedule(t)
        # print(T)  ### USED FOR DEBUGGING PURPOSES
        if T == 0 or problem.goal_test(current.state):
            return current.state
        neighbors = current.expand(problem)
        if not neighbors:
            return current.state
        next_choice = random.choice(neighbors)
        delta_e = problem.h(current) - problem.h(next_choice)
        if delta_e > 0 or search.probability(search.np.exp(delta_e / T)):
            current = next_choice
            
# Generates a random puzzle   
def random_puzzle():
    # Generate a list of integers from 0 to 8 and shuffle
    numbers = list(range(9))
    random.shuffle(numbers)
    
    # Return tuple of random numbers
    return tuple(numbers)

'''USED FOR TESTING PURPOSES

# puzzle = random_puzzle()
# # puzzle = (1, 7, 6, 5, 4, 2, 3, 0, 8)
# # puzzle = (1, 2, 3, 4, 5, 6, 7, 0, 8)
# print(puzzle)

# puzzle_problem = EightPuzzle(puzzle)
# current = search.Node(puzzle_problem.initial)


# output = simulated_annealing(puzzle_problem)
# print(output)

# print(puzzle_problem.value((1, 7, 6, 5, 4, 2, 3, 8, 0)))
'''

# Initialize successes, fails and times lists used to calculate those values respectively
successes = [0, 0, 0, 0]
fails = [0, 0, 0, 0]
times = [0, 0, 0, 0]

for _ in range(100):
    # Generate a random 8-puzzle
    puzzle = random_puzzle()
    
    # Create a problem object for each algorithm
    puzzle_problem_steepest = EightPuzzle(puzzle)
    puzzle_problem_first = EightPuzzle(puzzle)
    puzzle_problem_restart = EightPuzzle(puzzle)
    puzzle_problem_annealing = EightPuzzle(puzzle)
    
    # Measure the time taken to execute each algorithm
    start = time.time()
    steepest_output = steepest_hill_climbing(puzzle_problem_steepest)
    end = time.time()
    times[0] += (end - start)
    
    start = time.time()
    first_output = first_choice_hill_climbing(puzzle_problem_first)
    end = time.time()
    times[1] += (end - start)
    
    start = time.time()
    restart_output = random_restart_hill_climbing(puzzle_problem_restart, 1000)
    end = time.time()
    times[2] += (end - start)
    
    start = time.time()
    annealing_output = simulated_annealing(puzzle_problem_annealing)
    end = time.time()
    times[3] += (end - start)

    # Count the amount of successes and fails for each algorithm
    if puzzle_problem_steepest.goal_test(steepest_output):
        successes[0] += 1
    else:
        fails[0] += 1
    if puzzle_problem_first.goal_test(first_output):
        successes[1] += 1
    else:
        fails[1] += 1
    if puzzle_problem_restart.goal_test(restart_output):
        successes[2] += 1
    else:
        fails[2] += 1
    if puzzle_problem_annealing.goal_test(annealing_output):
        successes[3] += 1
    else:
        fails[3] += 1

# Print Statements
print("Out of 100 random 8-puzzles")
print("Steepest Hill Climbing")
print("     Successes:", successes[0])
print("     Fails:", fails[0])
print("     Time Taken:", times[0])
print("\nFirst Hill Climbing")
print("     Successes:", successes[1])
print("     Fails:", fails[1])
print("     Time Taken:", times[1])
print("\nRandom Restart Hill Climbing")
print("     Successes:", successes[2])
print("     Fails:", fails[2])
print("     Time Taken:", times[2])
print("\nSimulated Annealing")
print("     Successes:", successes[3])
print("     Fails:", fails[3])
print("     Time Taken:", times[3])