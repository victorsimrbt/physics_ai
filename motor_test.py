from joint import *
from genetic import *
import numpy as np
import random
Box()

arm_vec = Vec2d(0, 100)
p = Vec2d(400,250)

vs = [(-10, 40), (30, 40), (40, -40), (-40, -40),(100,-40)]
limbs = [arm_vec,arm_vec,arm_vec,arm_vec,arm_vec]
torso = Poly(p, vs)

for i in range(len(vs)):
    arm = Segment(p+vs[i], limbs[i])
    PivotJoint(torso.body, arm.body, vs[i], (0, 0))
    SimpleMotor(torso.body,arm.body,random.randint(1,5))
App(draw=True).run()