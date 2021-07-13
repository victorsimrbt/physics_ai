import numpy as np
import random
import numpy as np
from IPython.display import clear_output
from ragdoll_model import *

num_joints = 5

class genetic_algorithm:
        
    def execute(self,pop_size,generations,threshold,fitness):
        class Agent:
            def __init__(self):
                self.body_net = body_net(num_joints)
                self.motor_net = motor_net(num_joints)
                self.fitness = 0
            def apply_body_weights(self,weights):
                self.body_net.set_weights(weights)
            def apply_motor_weights(self,weights):
                self.motor_net.set_weights(weights)
            def __str__(self):
                    return 'Loss: ' + str(self.fitness)
        
        def generate_agents(population):
            return [Agent() for _ in range(population)]
        
        def selection(agents):
            agents = sorted(agents, key=lambda agent: agent.fitness, reverse=True)
            print('\n'.join(map(str, agents)))
            agents = agents[:int(0.2 * len(agents))]
            return agents
        
        def unflatten(flattened,shapes):
            newarray = []
            index = 0
            for shape in shapes:
                size = np.product(shape)
                newarray.append(flattened[index : index + size].reshape(shape))
                index += size
            return np.array(newarray)
        
        def crossover(agents,pop_size):
            offspring = []
            for _ in range((pop_size - len(agents)) // 2):
                parent1 = random.choice(agents)
                parent2 = random.choice(agents)
                child1 = Agent()
                child2 = Agent()
                
                
                body_shapes = [a.shape for a in parent1.body_net.get_weights()]
                body_genes1 = np.concatenate([a.flatten() for a in parent1.body_net.get_weights()])
                body_genes2 = np.concatenate([a.flatten() for a in parent2.body_net.get_weights()])
                body_split = random.randint(0,len(body_genes1)-1)


                body_child1_genes = np.array(body_genes1[0:body_split].tolist() + body_genes2[body_split:].tolist())
                body_child2_genes = np.array(body_genes1[0:body_split].tolist() + body_genes2[body_split:].tolist())
                body_child1_genes = unflatten(body_child1_genes,body_shapes)
                body_child2_genes = unflatten(body_child2_genes,body_shapes)

                motor_shapes = [a.shape for a in parent1.motor_net.get_weights()]
                motor_genes1 = np.concatenate([a.flatten() for a in parent1.motor_net.get_weights()])
                motor_genes2 = np.concatenate([a.flatten() for a in parent2.motor_net.get_weights()])
                motor_split = random.randint(0,len(motor_genes1)-1)

                motor_child1_genes = np.array(motor_genes1[0:motor_split].tolist() + motor_genes2[motor_split:].tolist())
                motor_child2_genes = np.array(motor_genes1[0:motor_split].tolist() + motor_genes2[motor_split:].tolist())
                motor_child1_genes = unflatten(motor_child1_genes,motor_shapes)
                motor_child2_genes = unflatten(motor_child2_genes,motor_shapes)
            
                
                child1.apply_body_weights(list(body_child1_genes))
                child2.apply_body_weights(list(body_child2_genes))

                child1.apply_motor_weights(list(motor_child1_genes))
                child2.apply_motor_weights(list(motor_child2_genes))
                
                offspring.append(child1)
                offspring.append(child2)
            agents.extend(offspring)
            return agents
        
        def mutation(agents):
            for agent in agents:
                if random.uniform(0.0, 1.0) <= 0.1:
                    weights = agent.body_net.get_weights()
                    shapes = [a.shape for a in weights]

                    flattened = np.concatenate([a.flatten() for a in weights])
                    randint = random.randint(0,len(flattened)-1)
                    flattened[randint] = np.random.randn()

                    newarray = unflatten(flattened,shapes)
                    agent.apply_body_weights(newarray)

                    weights = agent.motor_net.get_weights()
                    shapes = [a.shape for a in weights]

                    flattened = np.concatenate([a.flatten() for a in weights])
                    randint = random.randint(0,len(flattened)-1)
                    flattened[randint] = np.random.randn()

                    newarray = unflatten(flattened,shapes)
                    agent.apply_motor_weights(newarray)
            return agents
        
        loss = []
        for i in range(generations):
            print('Generation',str(i),':')
            agents = generate_agents(pop_size)
            agents = fitness(agents)
            agents = selection(agents)
            agents = crossover(agents,pop_size)
            agents = mutation(agents)
            agents = fitness(agents)
            loss.append(agents[0].fitness)
            if any(agent.fitness > threshold for agent in agents):
                print('Threshold met at generation '+str(i)+' !')
                
            if i % 100:
                clear_output()
                
        return agents[0],loss