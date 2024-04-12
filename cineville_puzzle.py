import numpy as np
import random
import numpy as np
from collections import deque

cities = [
    "Alkmaar", "Amsterdam", "Amersfoort", "Apeldoorn", "Arnhem", "Breda", "Bussum", "Castricum",
    "Delft", "Den Bosch", "Den Haag", "Deventer", "Dordrecht", "Eindhoven", "Enschede", "Enkhuizen", 
    "Groningen", "Goes", "Haarlem", "Helmond", "Hilversum", "Hoorn", "Houten", "IJmuiden", "Leeuwarden",
    "Leiden", "Maastricht", "Middelburg", "Nijmegen", "Oostburg", "Rotterdam", "Schiedam", "Utrecht",
    "Voorschoten", "Wageningen", "Zaandam", "Zierikzee", "Zevenaar", "Zwolle"
]

data = """
,Alkmaar,Amsterdam,Amersfoort,Apeldoorn,Arnhem,Breda,Bussum,Castricum,Delft,Den Bosch,Den Haag,Deventer,Dordrecht,Eindhoven,Enschede,Enkhuizen,Groningen,Goes,Haarlem,Helmond,Hilversum,Hoorn,Houten,IJmuiden,Leeuwarden,Leiden,Maastricht,Middelburg,Nijmegen,Oostburg,Rotterdam,Schiedam,Utrecht,Voorschoten,Wageningen,Zaandam,Zevenaar,Zierikzee,Zwolle
Alkmaar,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
Amsterdam,,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
Amersfoort,,,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
Apeldoorn,,,43,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
Arnhem,,,44,27,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
Breda,,,,,,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
Bussum,,23,23,62,,,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
Castricum,12,,,,,,,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
Delft,,,,,,,,,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
Den Bosch,,,,,,45,,,,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
Den Haag,,,,,,,,,8,,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,
Deventer,,,,15,39,,,,,,,0,,,,,,,,,,,,,,,,,,,,,,,,,,,
Dordrecht,,,,,,34,,,36,55,,,0,,,,,,,,,,,,,,,,,,,,,,,,,,
Eindhoven,,,,,,57,,,,35,,,,0,,,,,,,,,,,,,,,,,,,,,,,,,
Enschede,,,,70,80,,,,,,,56,,,0,,,,,,,,,,,,,,,,,,,,,,,,
Enkhuizen,41,,82,88,,,71,,,,,,,,,0,,,,,,,,,,,,,,,,,,,,,,,
Groningen,,,,,,,,,,,,,,,132,126,0,,,,,,,,,,,,,,,,,,,,,,
Goes,,,,,,74,,,,,,,,,,,,0,,,,,,,,,,,,,,,,,,,,,
Haarlem,,19,,,,,,20,,,,,,,,,,,0,,,,,,,,,,,,,,,,,,,,
Helmond,,,,,,68,,,,38,,,,17,,,,,,0,,,,,,,,,,,,,,,,,,,
Hilversum,,26,19,60,,,5,,,,,,,,,,,,,,0,,,,,,,,,,,,,,,,,,
Hoorn,24,36,,,,,,34,,,,,,,,18,,,,,,0,,,,,,,,,,,,,,,,,
Houten,,,27,,,71,,,,41,,,50,,,,,,,,,,0,,,,,,,,,,,,,,,,
IJmuiden,25,,,,,,,13,69,,,,,,,,,,10,,,,,0,,,,,,,,,,,,,,,
Leeuwarden,,,,,,,,,,,,,,,,71,57,,,,,,,,0,,,,,,,,,,,,,,
Leiden,,41,,,,,,,,,,,50,,,,,,29,,53,,63,,,0,,,,,,,,,,,,,
Maastricht,,,,,,116,,,,,,,,88,,,,,,82,,,,,,,0,,,,,,,,,,,,
Middelburg,,,,,,97,,,,,,,,,,,,22,,,,,,,,,,0,,,,,,,,,,,
Nijmegen,,,,,18,,,,,45,,,,62,,,,,,,,,56,,,,,,0,,,,,,,,,,
Oostburg,,,,,,119,,,,,,,,178,,,,44,,,,,,,,,,24,,0,,,,,,,,,
Rotterdam,,,,,,,,,14,,,,22,,,,,,,,,,56,,,,,,,,0,,,,,,,,
Schiedam,,,,,,,,,12,,,,,,,,,,,,,,,,,,,,,,6,0,,,,,,,
Utrecht,,39,20,,,,,,61,,,,,,,,,,,,17,,13,,,53,,,,,55,,0,,,,,,
Voorschoten,,,,,,,,,15,,12,,,,,,,,,,,,,,,6,,,,,26,,,0,,,,,
Wageningen,,,34,40,17,,,,,48,,,,,,,,,,,,,38,,,,,,28,,,,44,,0,,,,
Zaandam,26,12,,,,,,18,,,,,,,,,,,17,,,32,,18,,,,,,,,,,,,0,,,
Zevenaar,,,,38,15,,,,,,,44,,,72,,,,,,,,,,,,,,23,,,,,,32,,0,,
Zierikzee,,,,,,71,,,70,,,,68,,,,,20,,,,,,,,,,36,,60,,67,,,,,,0,
Zwolle,,,69,38,,,79,,,,,31,,,70,65,95,,,,,,,,,,,,,,,,,,,,,,0
"""

# Convert data from csv string above to np array
rows = data.strip().split('\n')
cities = [row.split(',')[0] for row in rows[1:]]
distances = [[int(x) if x != '' else np.inf for x in row.split(',')[1:]] for row in rows[1:]]
distance_matrix = np.array(distances)

"""
print("Cities:", cities)
print("Distance array:")
print(distance_matrix)
"""


def simulated_annealing(cost_function, initial_solution, temperature, cooling_rate):
    current_solution = initial_solution
    current_cost = cost_function(current_solution)

    while temperature > 0.1:
        new_solution = current_solution.copy()
        index = random.randint(1, len(new_solution) - 2)  # Avoid changing start/end cities
        new_index = random.randint(1, len(new_solution) - 2)
        
        if index != new_index:
            value_to_move = new_solution.pop(index)
            new_solution.insert(new_index, value_to_move)

        new_cost = cost_function(new_solution)
        
        if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / temperature):
            current_solution = new_solution
            current_cost = new_cost

        temperature *= cooling_rate
        print("cost:", current_cost)

    return current_solution

def greedy_exhaustive_search(cost_function, initial_solution):
    best_solution = initial_solution
    current_solution = initial_solution
    best_cost = cost_function(best_solution)
    n = len(initial_solution)
    
    for depth in range(100):
        current_solution = best_solution.copy()
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                if i != j:
                    #Try inserting somewhere else
                    new_solution = current_solution.copy()
                    value_to_move = new_solution.pop(i)
                    new_solution.insert(j, value_to_move)
                    new_solution.insert(i, j)
                    new_cost = cost_function(new_solution)
                    if new_cost < best_cost:
                        best_solution = new_solution.copy()
                        best_cost = new_cost
                    #Try swapping two
                    new_solution = current_solution.copy()
                    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
                    new_cost = cost_function(new_solution)
                    if new_cost < best_cost:
                        best_solution = new_solution.copy()
                        best_cost = new_cost
        #tsp_cost(best_solution, verbose=True)                                
        print("cost:",best_cost)
    return best_solution

def bfs_search(cost_function, initial_solution, max_depth=10):
    best_solution = initial_solution
    best_cost = cost_function(best_solution)
    n = len(initial_solution)
    
    # Create a queue for BFS
    queue = deque([(initial_solution, best_cost, 0)])  # Include depth in the queue
    visited = set([tuple(initial_solution)])
    
    while queue:
        current_solution, current_cost, depth = queue.popleft()

        # Check if depth exceeds the maximum depth
        if depth >= max_depth:
            continue

        # Generate all possible neighbors of the current solution
        for i in range(1, n - 1):
            for j in range(i + 1, n - 1):
                # Try inserting somewhere else
                new_solution = current_solution.copy()
                value_to_move = new_solution.pop(i)
                new_solution.insert(j, value_to_move)
                new_cost = cost_function(new_solution)
                if new_cost < best_cost and tuple(new_solution) not in visited:
                    best_solution = new_solution.copy()
                    best_cost = new_cost
                    print("insert")
                    print(new_cost)
                    print(new_solution)
                if tuple(new_solution) not in visited and new_cost < 9000:
                    queue.append((new_solution, new_cost, depth + 1))
                    visited.add(tuple(new_solution))
                
                # Try swapping two
                new_solution = current_solution.copy()
                new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
                new_cost = cost_function(new_solution)
                if new_cost < best_cost and tuple(new_solution) not in visited:
                    best_solution = new_solution.copy()
                    best_cost = new_cost
                    print("switch")
                    print(new_cost)
                    print(new_solution)
                if tuple(new_solution) not in visited and new_cost < 9000:
                    queue.append((new_solution, new_cost, depth + 1))
                    visited.add(tuple(new_solution))

    return best_solution

# Cost function for path
def tsp_cost(path, verbose = False):
    cost = 0
    for i in range(len(path) - 1):
        distance = distance_matrix[max(path[i], path[i+1])][min(path[i], path[i+1])]
        if verbose: print(cities[path[i]],"to", cities[path[i+1]], distance)
        cost += distance
    if verbose: print("Total distance:", cost)
    return cost

annealing_solution = [24, 16, 38, 14, 11, 3, 4, 36, 28, 34, 2, 22, 32, 20, 6, 1, 35, 21, 15, 0, 7, 23, 18, 25, 33, 10, 8, 31, 30, 12, 37, 17, 27, 29, 5, 9, 13, 19, 26]
current_frontrunner = [24, 16, 38, 14, 11, 3, 4, 36, 28, 34, 22, 32, 2, 20, 6, 1, 35, 21, 15, 0, 7, 23, 18, 25, 33, 10, 8, 31, 30, 12, 37, 27, 29, 17, 5, 9, 13, 19, 26]
"""
tsp_cost(annealing_solution, verbose=True)
tsp_cost(current_frontrunner, verbose=True)

random_solution = annealing_solution.copy()  
random.shuffle(random_solution)  
distance_matrix[distance_matrix == np.inf] = 9999
final_solution = bfs_search(tsp_cost, current_frontrunner)
tsp_cost(random_solution, verbose=True)
tsp_cost(final_solution, verbose=True)
"""

for i in current_frontrunner:
    print(cities[i],end="")
    print(", ",end="")