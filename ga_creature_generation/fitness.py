from joint import *
import cv2
def fitness(agents):
    counter = 0
    for agent in agents:
        counter += 1
        
        body_range = 100
        motor_range = 5
        arm_range = 200
        
        num_joints = 5
        len_steps = 100

        Box()
        arm_vec = Vec2d(0, 100)
        p = Vec2d(400,250)

        latent_data = np.random.randn(num_joints,2)
        motor_data = np.random.randn(num_joints)
        arm_data = np.random.randn(num_joints)
        
        vs = agent.body_net.predict(latent_data)
        vs = (vs*body_range).astype(int)
        true_vs = []
        for i in range(len(vs)):
            true_vs.append((vs[i][0],vs[i][1]))
        torso = Poly(p, true_vs)

        motor_values = agent.motor_net.predict(motor_data)
        motor_values = (motor_values*motor_range).astype(int)
        
        arm_values = agent.arm_net.predict(arm_data)
        limbs = []
        for value in arm_values:
            limbs.append(Vec2d(0,value*arm_range))
            

        for i in range(len(vs)):
            arm = Segment(p+vs[i], limbs[i])
            PivotJoint(torso.body, arm.body, true_vs[i], (0, 0))
            SimpleMotor(b0,arm.body,motor_values[i])
            
        if counter % 2 == 0:
            val = 'A'
        else:
            val = 'B'
        gif_name = 'joint'+val+'.gif'
        
        app = App(gif_path = gif_name)
        app.run(steps = len_steps)
        app.make_gif()

        area = cv2.contourArea(vs)
        agent.fitness = abs(torso.body.position[0])/area

    return agents