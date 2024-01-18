import numpy as np
import pandas as pd
import random
import tqdm
import csv
data = pd.read_csv("train100.csv")
data["Date"] = pd.to_datetime(data["Date"])
data = data.set_index('Date')
# data["Date"] = data["Date"].astype(np.int64)
flow_data = data["Flow"].values
data["Flow"] = (data["Flow"] - data["Flow"].min()) / (data["Flow"].max() - data["Flow"].min())
train_data = data[:-100]
test_data = data[-100:]
population_size = 100
num_generations = 100
mutation_rate = 0.01
def generate_population(population_size):
    population = []

    for _ in range(population_size):
        chromosome = []

        # 随机生成100个0到1之间的数作为染色体
        for _ in range(100):
            chromosome.append(random.uniform(0, 1))

        population.append(chromosome)

    return population
def evaluate_fitness(chromosome):
    # 根据染色体预测接下来100天的客流量
    predicted_flow = chromosome[:100]

    # 计算预测客流量与实际客流量之间的均方根误差
    rmse = np.sqrt(np.mean((predicted_flow - flow_data[:100]) ** 2))

    return rmse
def select_parents(population):
    # 根据适应度进行选择
    fitness_scores = [evaluate_fitness(chromosome) for chromosome in population]
    parent_indices = np.argsort(fitness_scores)[:int(population_size / 2)]

    # 选择父代染色体
    parents = [population[i] for i in parent_indices]

    return parents


def crossover(parents):
    offspring = []

    for _ in range(population_size - len(parents)):
        # 随机选择两个父代染色体
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)

        # 从两个父代染色体中随机选择一定比例的基因
        num_genes = len(parent1)
        num_genes_to_keep = int(num_genes * (1 - mutation_rate))
        offspring_chromosome = random.sample(parent1, num_genes_to_keep)

        # 从另一个父代染色体中选择剩余的基因
        for gene in parent2:
            if gene not in offspring_chromosome:
                offspring_chromosome.append(gene)

        offspring.append(offspring_chromosome)

    return offspring


def mutate(offspring):
    for chromosome in offspring:
        # 对每个染色体的基因进行变异操作
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                # 在0到1之间随机生成一个新的基因
                chromosome[i] = random.uniform(0, 3000)

    return offspring
population = generate_population(population_size)

for _ in tqdm.tqdm(range(num_generations)):
    # 选择父代染色体
    parents = select_parents(population)

    # 生成子代染色体
    offspring = crossover(parents)

    # 变异子代染色体
    offspring = mutate(offspring)

    # 合并父代和子代染色体
    population = parents + offspring

# 选择适应度最高的染色体作为最终结果
best_chromosome = min(population, key=evaluate_fitness)

# 根据最优染色体预测接下来100天的客流量
predicted_flow = best_chromosome[:100]
# print(len(predicted_flow))
# 反归一化预测结果
predicted_flow = predicted_flow * (np.max(flow_data) - np.min(flow_data)) + np.min(flow_data)
# 打印预测结果
# print(len(predicted_flow))
# print(predicted_flow)
future_dates = pd.date_range(start=data.index[-1] + pd.DateOffset(1), periods=100)
future_dates = pd.DataFrame(future_dates, columns=['Date'])
future_dates['Date'] = pd.to_datetime(future_dates['Date'])
future_dates = future_dates.set_index('Date')
# 构建预测结果DataFrame
# print(predicted_flow.shape)
future_dates['Flow'] = predicted_flow[:100].astype(int)
#将预测结果保存为csv文件
future_dates.to_csv('future_flow.csv')
#将预测的结果画出来

# 打印预测结果
print(future_dates)

