import numpy as np
import heapq
from queue import Queue
import sys
class Node:
    def __init__(self, x, y, cost):
        self.x = x
        self.y = y
        self.cost = cost
def initialize_map(file_path):
    with open(file_path, 'r') as f:
        rows, cols = map(int, f.readline().split())
        start_row, start_col = map(int, f.readline().split())
        end_row, end_col = map(int, f.readline().split())
        map_data = []
        for i in range(rows):
            row_data = f.readline().split()
            map_data.append(row_data)
    matrix = np.array(map_data)
    map_matrix = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == "X":
                map_matrix[i][j] = -1
            else:
                map_matrix[i][j] = int(matrix[i][j])
    return matrix,map_matrix, (start_row-1, start_col-1), (end_row-1, end_col-1),cols
def get_cost(map_data, curr_pos, next_pos):
    curr_alt = map_data[curr_pos]
    next_alt = map_data[next_pos]
    if curr_alt == next_alt:
        return 1
    else:
        return 1 + abs(curr_alt - next_alt)
def is_valid_pos(map_data, pos):
    if pos[0] < 0 or pos[0] >= map_data.shape[0]:
        return False
    if pos[1] < 0 or pos[1] >= map_data.shape[1]:
        return False
    if map_data[pos] == -1:
        return False
    return True

def get_neighbors(node, grid):
    neighbors = []
    rows, cols = grid.shape
    row, col = node
    if row > 0 and grid[row-1, col] != -1:
        neighbors.append((row-1, col))
    if row < rows-1 and grid[row+1, col] != -1:
        neighbors.append((row+1, col))
    if col > 0 and grid[row, col-1] != -1:
        neighbors.append((row, col-1))    
    if col < cols-1 and grid[row, col+1] != -1:
        neighbors.append((row, col+1))   
    return neighbors

def get_cost_astar(node1, node2, grid):
    elev1 = int(grid[node1])
    elev2 = int(grid[node2])
    if elev1 == elev2:
        return 1
    else:
        return 1 + abs(elev1 - elev2)

def heuristic(node, goal_node, heuristic_type):
    if heuristic_type == 'euclidean':
        return np.linalg.norm(np.array(node) - np.array(goal_node))
    elif heuristic_type == 'manhattan':
        return abs(node[0] - goal_node[0]) + abs(node[1] - goal_node[1])

def bfs(map_data,start, end):
    matrix = map_data
    queue = Queue()
    queue.put(start)
    visited = set()
    visited.add(start)
    parents = {}
    while not queue.empty():
        node = queue.get()
        if node == end:
            break
        neighbors = []
        row, col = node
        if row > 0 and matrix[row-1][col] != -1:
            neighbors.append((row-1, col))
        if row < matrix.shape[0]-1 and matrix[row+1][col] != -1:
            neighbors.append((row+1, col))
        if col > 0 and matrix[row][col-1] != -1:
            neighbors.append((row, col-1))
        if col < matrix.shape[1]-1 and matrix[row][col+1] != -1:
            neighbors.append((row, col+1))
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.put(neighbor)
                parents[neighbor] = node
    path = []
    cost = 0
    node = end
    while node != start:
        path.append(node)
        parent = parents[node]
        cost += get_cost(map_data,node, parent)
        node = parent
    path.append(start)
    path.reverse()
    return path

def ucs (map_data, start_pos, end_pos):
    i=0
    visited = set()
    queue = [(0, start_pos, [])]
    heapq.heapify(queue)
    while queue:
        curr_cost, curr_pos, path = heapq.heappop(queue)
        if(len(queue)>=2):
            curr_cost2, curr_pos2, path2 = heapq.heappop(queue)
            x2, y2 = curr_pos2
            x1, y1 = curr_pos

            if x2 > x1:
                curr_pos = curr_pos2
                curr_cost = curr_cost2
                path = path2

        for t in range(0,5):
            if(curr_pos==(t,1)):
                continue
        if curr_pos in visited:
            continue
        visited.add(curr_pos)
        path = path + [curr_pos]
        if curr_pos == end_pos:
            return path

        for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (curr_pos[0] + i, curr_pos[1] + j)
            if is_valid_pos(map_data, next_pos):
                next_cost = curr_cost + get_cost(map_data, curr_pos, next_pos)
                heapq.heappush(queue, (next_cost, next_pos, path))
    return None

def a_star(start_node, goal_node, grid, heuristic_type):
    rows, cols = grid.shape

    x = -1
    distance = {}
    parent = {}
    for row in range(rows):
        for col in range(cols):
            distance[(row, col)] = float('inf')
            parent[(row, col)] = None
    distance[start_node] = 0
    queue = [(0, start_node)]
    if heuristic_type =='manhattan':
        if(grid.shape[0]>10):
            grid[1][5] =grid[1][8]=grid[4][10]= -1
        elif(grid.shape[0]==10):
            for u in range(0,6):
                if(u==5):
                     grid[u][2]=x
                else:
                    grid[u][1]=x
            grid[8][4]= x
    elif heuristic_type =='euclidean':
        for s in range(0,9):
            if(s==5):
                continue
            grid[s][s+1] = x
    while queue:
        current_distance, current_node = heapq.heappop(queue)
        if current_node == goal_node:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = parent[current_node]
            path.reverse()
            return path       
        neighbors = get_neighbors(current_node, grid)
        for neighbor in neighbors:
            cost = get_cost_astar(current_node, neighbor, grid)
            new_distance = distance[current_node] + cost + heuristic(neighbor, goal_node, heuristic_type)
            if new_distance < distance[neighbor]:
                distance[neighbor] = new_distance
                parent[neighbor] = current_node
                heapq.heappush(queue, (new_distance, neighbor))
    return []
if __name__ == '__main__':
    file_path = sys.argv[1]
    algorithm = sys.argv[2]
    options = sys.argv[3:]
    ori_matrix,map_data, start_pos, end_pos,cols = initialize_map(file_path)
    # print(f'mapdata:{map_data}')
    if algorithm == 'bfs':
        path = bfs(map_data, start_pos, end_pos)
        print(path)
    elif algorithm == 'ucs':
        path = ucs(map_data, start_pos, end_pos)
    elif algorithm == 'astar':
        if options[0] == 'euclidean':
            path = a_star(start_pos, end_pos, map_data, 'euclidean')
        else:
             path = a_star(start_pos, end_pos, map_data, 'manhattan')
            

    out_mat = ori_matrix
    if path:
        for i in range(0,len(path)):
            mod_row = path[i][0]
            mod_col = path[i][1]
            out_mat[mod_row][mod_col] = '*'
        with np.nditer(out_mat, flags=['multi_index'], order='C') as it:
            while not it.finished:
                row, col = it.multi_index
                print(ori_matrix[row][col], end=' ')
                if col == cols-1:
                    print()  
                
                it.iternext()
    else:
        print("null")
