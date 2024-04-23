import search
import random
import time

# N Queens Problem class from the aimacode.
# Modified to use 8 as the default N and work with hill climb algorithms.
class NQueensProblem(search.Problem):
    """The problem of placing N queens on an NxN board with none attacking
    each other. A state is represented as an N-element array, where
    a value of r in the c-th entry means there is a queen at column c,
    row r, and a value of -1 means that the c-th column has not been
    filled in yet. We fill in columns left to right.
    >>> depth_first_tree_search(NQueensProblem(8))
    <Node (7, 3, 0, 2, 5, 1, 6, 4)>
    """

    def __init__(self, N=8):
        super().__init__(tuple([-1] * 8))
        self.N = N

    # Modified to be compatible with hill climb algorithms
    def actions(self, state):
        """In the leftmost empty column, try all non-conflicting rows."""
        if state[-1] != -1:
            actions = []
            for c in range(self.N):
                for r in range(self.N):
                    if r != state[c] and not self.conflicted(state, r, c, exclude=c):
                        actions.append((c, r))
        else:
            col = state.index(-1)
            actions = [row for row in range(self.N)
                    if not self.conflicted(state, row, col)]
        
        return actions

    # Modified to be compatible with hill climb algorithms
    def result(self, state, action):
        """Place the next queen at the given row."""
        new = list(state[:])
        if isinstance(action, int):
            c = state.index(-1)
            new[c] = action
        else:
            c, r = action
            new[c] = r
        return tuple(new)

    # Modified to be compatible with hill climb algorithms
    def conflicted(self, state, row, col, exclude=None):
        """Would placing a queen at (row, col) conflict with anything?"""
        return any(self.conflict(row, col, state[c], c)
                   for c in range(self.N) if state[c] != -1 and c != exclude and c != col) 

    def conflict(self, row1, col1, row2, col2):
        """Would putting two queens in (row1, col1) and (row2, col2) conflict?"""
        return (row1 == row2 or  # same row
                col1 == col2 or  # same column
                row1 - col1 == row2 - col2 or  # same \ diagonal
                row1 + col1 == row2 + col2)  # same / diagonal

    def goal_test(self, state):
        """Check if all columns filled, no conflicts."""
        if state[-1] == -1:
            return False
        return not any(self.conflicted(state, state[col], col)
                       for col in range(len(state)))

    def h(self, node):
        """Return number of conflicting queens for a given node"""
        num_conflicts = 0
        for (r1, c1) in enumerate(node.state):
            for (r2, c2) in enumerate(node.state):
                if (r1, c1) != (r2, c2):
                    num_conflicts += self.conflict(r1, c1, r2, c2)

        return num_conflicts
    
    # Use the already implemented heuristic function for the Queens problem in the aimacode repository
    def value(self, node):
        return self.h(node)
    
    # Function to set the initial state to a random one. Used for Random Restart Hill Climb
    def set_random_initial(self):
        # Generate a list of random integers from 0 to 7
        numbers = []
        for _ in range(self.N):
            numbers.append(random.randint(0, self.N-1))
        
        # Set initial state to a random state
        self.initial = tuple(numbers)
    
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
        neighbor = search.argmax_random_tie(neighbors, key=lambda node: problem.value(node))
        
        if problem.value(neighbor) >= problem.value(current):
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
            if problem.value(neighbor) < problem.value(current):
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
def random_restart_hill_climbing(problem, restarts):
    # Keeps track of the best encountered state across the restarts
    best_state = search.Node(problem.initial)
    
    for _ in range(restarts):
    # while True:              ### USED FOR DEBUGGING PURPOSES ###
        current = search.Node(problem.initial)
        
        while True:
            if problem.value(current) < problem.value(best_state):
                best_state = current
                if problem.goal_test(best_state.state):
                    return best_state.state
                    
            neighbors = current.expand(problem)
            if not neighbors:
                break
            neighbor = search.argmax_random_tie(neighbors, key=lambda node: problem.value(node))
            
            if problem.value(neighbor) >= problem.value(current):
                break
            current = neighbor
        
        problem.set_random_initial()    
    return best_state.state

# Schedule function from the aimacode repository.
# Modified to have a lam of 0.001 and limit of 5000.
def exp_schedule(k=20, lam=0.001, limit=5000):
    """One possible schedule function for simulated annealing"""
    return lambda t: (k * search.np.exp(-lam * t)) if t < limit else 0

# Simulated Annealing algorithm from the aimacode repository
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
        delta_e = problem.value(next_choice) - problem.value(current)
        if delta_e > 0 or search.probability(search.np.exp(delta_e / T)):
            current = next_choice

''' USED FOR TESTING PURPOSES
# def random_queens(length):
#     numbers = []
#     for _ in range(length):
#         numbers.append(random.randint(0, length-1))
#     return tuple(numbers)

# print(random_queens(8))
'''

# Initialize successes, fails and times lists used to calculate those values respectively         
successes = [0, 0, 0, 0]
fails = [0, 0, 0, 0]
times = [0, 0, 0, 0]

for _ in range(100):
    # Create a problem object for each algorithm
    queens_problem_steepest = NQueensProblem()
    queens_problem_first = NQueensProblem()
    queens_problem_restart = NQueensProblem()
    queens_problem_annealing = NQueensProblem()
    
    # Measure the time taken to execute each algorithm
    start = time.time()
    steepest_output = steepest_hill_climbing(queens_problem_steepest)
    end = time.time()
    times[0] += (end - start)
    
    start = time.time()
    first_output = first_choice_hill_climbing(queens_problem_first)
    end = time.time()
    times[1] += (end - start)
    
    start = time.time()
    restart_output = random_restart_hill_climbing(queens_problem_restart, 1000)
    end = time.time()
    times[2] += (end - start)
    
    start = time.time()
    annealing_output = simulated_annealing(queens_problem_annealing)
    end = time.time()
    times[3] += (end - start)
    
    # Count the amount of successes and fails for each algorithm
    if queens_problem_steepest.goal_test(steepest_output):
        successes[0] += 1
    else:
        fails[0] += 1
    if queens_problem_first.goal_test(first_output):
        successes[1] += 1
    else:
        fails[1] += 1
    if queens_problem_restart.goal_test(restart_output):
        successes[2] += 1
    else:
        fails[2] += 1
    if queens_problem_annealing.goal_test(annealing_output):
        successes[3] += 1
    else:
        fails[3] += 1

# Print Statements
print("Out of 100 random 8-queens")
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