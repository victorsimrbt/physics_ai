from joint import *
from genetic import *
from fitness import *

optimizer = genetic_algorithm()
fitness = fitness

top_agents, loss = optimizer.execute(10,100,1000,fitness)