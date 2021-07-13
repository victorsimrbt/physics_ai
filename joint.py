import pymunk
from pymunk.pygame_util import *
from pymunk.vec2d import Vec2d

import pygame
from pygame.locals import *

import math
from PIL import Image
import numpy as np

space = pymunk.Space()
space.gravity = 0, 900
b0 = space.static_body

size = w, h = 800, 500
fps = 30
steps = 10

BLACK = (0, 0, 0)
GRAY = (220, 220, 220)
WHITE = (255, 255, 255)

class PivotJoint:
    def __init__(self, b, b2, a=(0, 0), a2=(0, 0), collide=True):
        joint = pymunk.constraints.PinJoint(b, b2, a, a2)
        joint.collide_bodies = collide
        space.add(joint)


class SimpleMotor:
    def __init__(self, b, b2, rate):
        joint = pymunk.constraints.SimpleMotor(b, b2, rate)
        space.add(joint)


class Segment:
    def __init__(self, p0, v, radius=10):
        self.body = pymunk.Body()
        self.body.position = p0
        shape = pymunk.Segment(self.body, (0, 0), v, radius)
        shape.density = 0.1
        shape.elasticity = 0.5
        shape.filter = pymunk.ShapeFilter(group=1)
        shape.color = (0, 255, 0, 0)
        space.add(self.body, shape)


class Box:
    def __init__(self, p0=(0, 0), p1=(w, h), d=4):
        x0, y0 = p0
        x1, y1 = p1
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for i in range(4):
            segment = pymunk.Segment(
                space.static_body, pts[i], pts[(i+1) % 4], d)
            segment.elasticity = 1
            segment.friction = 0.5
            space.add(segment)


class Poly:
    def __init__(self, pos, vertices):
        self.body = pymunk.Body(1, 100)
        self.body.position = pos

        shape = pymunk.Poly(self.body, vertices)
        shape.filter = pymunk.ShapeFilter(group=1)
        shape.density = 0.01
        shape.elasticity = 0.5
        shape.color = (255, 0, 0, 0)
        space.add(self.body, shape)


class App:
    def __init__(self,draw=True,steps = 100):

        pygame.init()
        self.draw_bool = draw
        self.clock = pygame.time.Clock()
        if self.draw_bool == True:
            self.screen = pygame.display.set_mode(size)
            self.draw_options = DrawOptions(self.screen)
        self.running = True
        self.steps = steps
        self.gif = 0
        self.images = []

    def run(self,functions = None):
        comp_steps = 0
        while comp_steps < steps:
            for event in pygame.event.get():
                self.do_event(event)
            if functions:
                for function in functions:
                    function()
            if self.draw_bool:
                self.draw()
            self.clock.tick(fps)

            for i in range(steps):
                space.step(1/fps/steps)
            comp_steps += 1
        for body in space.bodies:
            space.remove(body)
        pygame.quit()

    def do_event(self, event):
        if event.type == QUIT:
            self.running = False

        if event.type == KEYDOWN:
            if event.key in (K_q, K_ESCAPE):
                self.running = False

            elif event.key == K_p:
                pygame.image.save(self.screen, 'joint.png')

            elif event.key == K_g:
                self.gif = 60

    def draw(self):
        self.screen.fill(GRAY)
        space.debug_draw(self.draw_options)
        pygame.display.update()

        text = f'fpg: {self.clock.get_fps():.1f}'
        pygame.display.set_caption(text)
        self.make_gif()

    def make_gif(self):
        if self.gif > 0:
            strFormat = 'RGBA'
            raw_str = pygame.image.tostring(self.screen, strFormat, False)
            image = Image.frombytes(
                strFormat, self.screen.get_size(), raw_str)
            self.images.append(image)
            self.gif -= 1
            if self.gif == 0:
                self.images[0].save('joint.gif',
                                    save_all=True, append_images=self.images[1:],
                                    optimize=True, duration=1000//fps, loop=0)
                self.images = []

