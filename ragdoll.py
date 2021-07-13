from joint import *
from genetic import *
import numpy as np
import cv2

def fitness(agents):
    for agent in agents:
        body_range = 100
        motors_range = 5
        num_joints = 5
        len_steps = 100

        Box()
        arm_vec = Vec2d(0, 100)
        p = Vec2d(400,250)

        latent_data = np.random.randn(num_joints,2)
        motor_data = np.random.randn(num_joints)
        vs = agent.body_net.predict(latent_data)
        vs = (vs*body_range).astype(int)
        true_vs = []
        for i in range(len(vs)):
            true_vs.append((vs[i][0],vs[i][1]))
        torso = Poly(p, true_vs)

        motor_values = agent.motor_net.predict(motor_data)
        motor_values = (motor_values*motors_range).astype(int)

        for i in range(len(vs)):
            arm = Segment(p+vs[i], arm_vec)
            PivotJoint(torso.body, arm.body, true_vs[i], (0, 0))
            SimpleMotor(b0,arm.body,motor_values[i])
        App(draw = False).run(steps = len_steps)

        area = cv2.contourArea(vs)
        agent.fitness = abs(torso.body.position[0])/area

    return agents

print('Genetic Algorithm')
optimizer = genetic_algorithm()

top_agents, loss = optimizer.execute(10,100,1000,fitness)