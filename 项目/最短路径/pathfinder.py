import numpy as np
import heapq
from queue import Queue
import sys
class Node:
    def __init__(self, x, y, cost):
        self.x = x
        self.y = y
        self.cost = cost
# 读取文件并初始化地图
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

# 计算两点间的花费
def get_cost(map_data, curr_pos, next_pos):
    # curr_alt = int(map_data[curr_pos])
    # next_alt = int(map_data[next_pos])
    curr_alt = map_data[curr_pos]
    next_alt = map_data[next_pos]
    if curr_alt == next_alt:
        return 1
    else:
        return 1 + abs(curr_alt - next_alt)

# 判断一个位置是否可行
def is_valid_pos(map_data, pos):
    if pos[0] < 0 or pos[0] >= map_data.shape[0]:
        return False
    if pos[1] < 0 or pos[1] >= map_data.shape[1]:
        return False
    if map_data[pos] == -1:
        return False
    return True

def get_neighbors(node, grid):
    """返回节点周围可到达的邻居"""
    neighbors = []
    rows, cols = grid.shape
    row, col = node
    
    # 上方邻居
    if row > 0 and grid[row-1, col] != -1:
        neighbors.append((row-1, col))
    
    # 下方邻居
    if row < rows-1 and grid[row+1, col] != -1:
        neighbors.append((row+1, col))
    
    # 左侧邻居
    if col > 0 and grid[row, col-1] != -1:
        neighbors.append((row, col-1))
    
    # 右侧邻居
    if col < cols-1 and grid[row, col+1] != -1:
        neighbors.append((row, col+1))
    
    return neighbors

def get_cost_astar(node1, node2, grid):
    """返回节点之间的代价"""
    elev1 = int(grid[node1])
    elev2 = int(grid[node2])
    if elev1 == elev2:
        return 1
    else:
        return 1 + abs(elev1 - elev2)

def heuristic(node, goal_node, heuristic_type):
    """返回节点到目标节点的启发式距离"""
    if heuristic_type == 'euclidean':
        return np.linalg.norm(np.array(node) - np.array(goal_node))
    elif heuristic_type == 'manhattan':
        return abs(node[0] - goal_node[0]) + abs(node[1] - goal_node[1])
    else:
        raise ValueError('Invalid heuristic type')

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

    # 从终点开始回溯路径
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
# 寻找从起点到终点的最短路径
def ucs (map_data, start_pos, end_pos):

    visited = set()
    queue = [(0, start_pos, [])]
    heapq.heapify(queue)
    while queue:
        curr_cost, curr_pos, path = heapq.heappop(queue)
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
    """使用A*算法寻找从起点到终点的最短路径"""
    rows, cols = grid.shape
    
    # 创建空的距离和父节点字典
    distance = {}
    parent = {}
    for row in range(rows):
        for col in range(cols):
            distance[(row, col)] = float('inf')
            parent[(row, col)] = None
    
    # 初始化起点距离为0
    distance[start_node] = 0
    
    # 创建一个优先队列来存储节点
    queue = [(0, start_node)]
    
    # A*主循环
    while queue:
        # 取出距离最小的节点
        current_distance, current_node = heapq.heappop(queue)
        # 如果当前节点为目标节点，则返回路径
        if current_node == goal_node:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = parent[current_node]
            path.reverse()
            return path
        
        # 获取当前节点周围的邻居
        neighbors = get_neighbors(current_node, grid)
        
        # 更新邻居节点的距离和父节点
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
    if algorithm == 'bfs':
        path = bfs(map_data, start_pos, end_pos)
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
            mod_row = path[i][1]
            mod_col = path[i][0]
            out_mat[mod_row][mod_col] = '*'
        with np.nditer(out_mat, flags=['multi_index'], order='C') as it:
            while not it.finished:
                row, col = it.multi_index
                print(ori_matrix[row][col], end=' ')
                if col == cols-1:
                    print()  # 换行
                
                it.iternext()
                
                
    else:
        print("null")
