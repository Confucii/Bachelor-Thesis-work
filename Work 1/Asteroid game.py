import random

import numpy as np

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 10)
plt.ion()


def rotation_mat(degrees):
    theta = np.radians(degrees)
    sin = np.sin(theta)
    cos = np.cos(theta)
    R = np.array([
        [cos, -sin, 0],
        [sin, cos, 0],
        [0, 0, 1]
    ])
    return R


def translation_mat(dx, dy):
    T = np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1]
    ])
    return T


def scale_mat(sx, sy):
    S = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])
    return S


def skew_mat(xs, ys):
    S = np.array([
        [1, ys, 0],
        [xs, 1, 0],
        [0, 0, 1]
    ])
    return S


def dim(arr):
    if isinstance(arr, np.ndarray) or isinstance(arr, list):
        return 1 + dim(arr[0])
    return 0


def dot(X, Y):
    Z = list()
    dimx = dim(X)
    dimy = dim(Y)
    if dimx == 1 and dimy == 1: #ndim can be used instead of dim(arr)
        if len(X) != len(Y):
            return "Impossible dot product"
        temp = 0
        for i in range(len(X)):
            temp += X[i] * Y[i]
        Z.append(temp)
    elif dimy == 1: #ndim can be used instead of dim(arr)
        if len(X[0]) != len(Y):
            return "Impossible dot product"
        for ix in range(len(X)):
            temp = 0
            for iy in range(len(Y)):
                temp += X[ix][iy] * Y[iy]
            Z.append(temp)
    elif dimx == 1: #ndim can be used instead of dim(arr)
        if len(X) != len(Y):
            return "Impossible dot product"
        for jy in range(len(Y[0])):
            temp = 0
            for ix in range(len(X)):
                temp += X[ix] * Y[ix][jy]
            Z.append(temp)
    else:
        if len(X[0]) != len(Y):
            return "Impossible dot product"
        for ix in range(len(X)):
            templist = list()
            for jy in range(len(Y[0])):
                temp = 0
                for iy in range(len(Y)):
                    temp += X[ix][iy] * Y[iy][jy]
                templist.append(temp)
            Z.append(templist)
    Y = np.array(Z)
    return Y


def vec2d_to_vec3d(vec2):
    I = np.array([
        [1, 0],
        [0, 1],
        [0, 0]
    ])
    vec3 = dot(I, vec2) + np.array([0, 0, 1])
    return vec3


def vec3d_to_vec2d(vec3):
    I = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ])
    vec2 = dot(I, vec3)
    return vec2


vec2 = np.array([1.0, 0.0])
vec3 = vec2d_to_vec3d(vec2)
vec2 = vec3d_to_vec2d(vec3)

A = np.array([
    [1, 2, 3, 4],
    [1, 2, 3, 4]
             ])
B = np.array([
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]
])
C = dot(A, B)
print(C)


class Character(object):
    def __init__(self):
        super().__init__()
        self.__angle = np.random.random() * np.pi

        self.geometry = []
        self.color = 'r'

        self.pos = np.array([np.random.uniform(-6, 7), np.random.uniform(-6, 7)])
        self.speed = 0.1
        self.dir_init = np.array([0.0, 1.0])
        self.dir = np.array(self.dir_init)
        self.r = 0.1

        self.C = np.identity(3)
        self.T = translation_mat(self.pos[0], self.pos[1])
        self.R = rotation_mat(self.__angle)
        self.S = np.identity(3)

        self.generate_geometry()

    def set_pos(self, posx, posy):
        self.pos[0] = posx
        self.pos[1] = posy
        self.T = translation_mat(self.pos[0], self.pos[1])

    def set_angle(self, angle):
        self.__angle = angle
        self.R = rotation_mat(self.__angle)

    def get_angle(self):
        return self.__angle

    def draw(self):
        x_data = []
        y_data = []
        self.C = dot(self.T, self.R)
        self.C = dot(self.C, self.S)
        for vec2 in self.geometry:
            vec3d = vec2d_to_vec3d(vec2)
            vec3d = dot(self.C, vec3d)
            vec2 = vec3d_to_vec2d(vec3d)
            x_data.append(vec2[0])
            y_data.append(vec2[1])
        plt.plot(x_data, y_data, color=self.color)

    def generate_geometry(self):
        pass


class Player(Character):
    def __init__(self):
        super().__init__()
        self.S = scale_mat(0.5, 1)

    def generate_geometry(self):
        self.geometry = np.array([
            [-1, 0],
            [1, 0],
            [0, 1],
            [-1, 0]
        ])
        # before the translation matrix I have used the code below to generate in relation to new position but
        # with this it rotates in relation to [0, 0]
        '''self.geometry = np.array([ 
            [self.pos[0], self.pos[1] + 0.1],
            [self.pos[0] + 0.5, self.pos[1] - 0.5],
            [self.pos[0] - 0.5, self.pos[1] - 0.5],
            [self.pos[0], self.pos[1] + 0.1]
        ])'''


class Asteroid(Character):
    def __init__(self):
        super().__init__()
        self.color = (np.random.random(), np.random.random(), np.random.random())
        self.pos = np.array([np.random.uniform(-9, 10), np.random.uniform(-9, 10)])
        self.T = translation_mat(self.pos[0], self.pos[1])
        self.S = skew_mat(np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5))
        self.speed = np.random.uniform(0.05, 0.2)
        self.dir = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)])

    def generate_geometry(self):
        self.r = np.random.uniform(0.2, 0.9)
        for i in range(361):
            if i % 20 == 0:
                self.geometry.append([self.r * np.cos(np.radians(i)) + np.random.uniform(0, 0.07), self.r * np.sin(np.radians(i)) + np.random.uniform(0, 0.07)])
        self.geometry = np.array(self.geometry)

    def set_pos(self, posx, posy):
        self.pos[0] = posx
        self.pos[1] = posy
        self.T = translation_mat(self.pos[0], self.pos[1])

    def draw(self):
        x_data = []
        y_data = []
        self.C = dot(self.T, self.S)
        for vec2 in self.geometry:
            vec3d = vec2d_to_vec3d(vec2)
            vec3d = dot(self.C, vec3d)
            vec2 = vec3d_to_vec2d(vec3d)
            x_data.append(vec2[0])
            y_data.append(vec2[1])
        plt.plot(x_data, y_data, c=self.color)


class Rocket(Character):
    def __init__(self):
        super().__init__()
        self.color = (np.random.random(), np.random.random(), np.random.random())
        self.speed = 0.2

    def get_player_data(self, player):
        self.pos = np.array(player.pos)
        self.R = player.R
        self.dir = player.dir

    def generate_geometry(self):
        self.r = 0.3
        self.geometry = np.array([
            [0, 1],
            [0, 1.2]
        ])

    def set_pos(self, posx, posy):
        self.pos[0] = posx
        self.pos[1] = posy
        self.T = translation_mat(self.pos[0], self.pos[1])

    def draw(self):
        x_data = []
        y_data = []
        self.C = dot(self.T, self.R)
        for vec2 in self.geometry:
            vec3d = vec2d_to_vec3d(vec2)
            vec3d = dot(self.C, vec3d)
            vec2 = vec3d_to_vec2d(vec3d)
            x_data.append(vec2[0])
            y_data.append(vec2[1])
        plt.plot(x_data, y_data, c=self.color)


characters = list()
player = Player()
characters.append(player)
asteroid_num = 0

for i in range(10):
    characters.append(Asteroid())
    asteroid_num += 1

is_running = True


def on_press(event):
    global is_running, player
    if event.key == 'escape':
        is_running = False
    elif event.key == 'left':
        player.set_angle(player.get_angle() + 5)
    elif event.key == 'right':
        player.set_angle(player.get_angle() - 5)
    elif event.key == ' ':
        characters.append(Rocket())
        characters[-1].get_player_data(player)


fig, _ = plt.subplots()
fig.canvas.mpl_connect('key_press_event', on_press)
while is_running:
    player.dir = player.dir_init
    player.dir = vec3d_to_vec2d(dot(player.R, vec2d_to_vec3d(player.dir)))
    player.set_pos(player.pos[0] + player.speed * player.dir[0], player.pos[1] + player.speed * player.dir[1])

    plt.clf()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    for entity in characters:
        for entity_check in characters:
            if entity is not entity_check:
                if ((entity.pos[0] - entity_check.pos[0]) ** 2 + (entity.pos[1] - entity_check.pos[1]) ** 2)**(1 / 2) <= entity.r or ((entity.pos[0] - entity_check.pos[0]) ** 2 + (entity.pos[1] - entity_check.pos[1]) ** 2)**(1 / 2) <= entity_check.r:
                    if isinstance(entity_check, Player) and isinstance(entity, Asteroid):
                        exit()
                    elif isinstance(entity_check, Rocket) and isinstance(entity, Asteroid):
                        characters.remove(entity)
                        characters.remove(entity_check)
                        asteroid_num -= 1

        if isinstance(entity, Rocket):
            entity.set_pos(entity.pos[0] + entity.speed * entity.dir[0], entity.pos[1] + entity.speed * entity.dir[1])
            if entity.pos[0] >= 10 or entity.pos[0] <= -10 or entity.pos[1] >= 10 or entity.pos[1] <= -10:
                characters.remove(entity)
        if isinstance(entity, Asteroid):
            entity.set_pos(entity.pos[0] + entity.speed * entity.dir[0], entity.pos[1] + entity.speed * entity.dir[1])
            if entity.pos[0] >= 10 or entity.pos[0] <= -10:
                entity.dir = np.array([-entity.dir[0], entity.dir[1]])
            elif entity.pos[1] >= 10 or entity.pos[1] <= -10:
                entity.dir = np.array([entity.dir[0], -entity.dir[1]])
        if isinstance(entity, Player):
            if entity.pos[0] >= 10 or entity.pos[0] <= -10 or entity.pos[1] >= 10 or entity.pos[1] <= -10:
                exit()
        entity.draw()
        plt.title(f"Angle: {player.get_angle() % 360}, Asteroids left: {asteroid_num}")

    plt.draw()
    plt.pause(1e-3)
