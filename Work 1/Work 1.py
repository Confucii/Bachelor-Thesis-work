import numpy as np
import sklearn.datasets
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 10)
plt.ion()
import torch

# Task 1


def power_func(x, n):
    result = 1
    if n == 0:
        return result
    elif n > 0:
        for i in range(n):
            result *= x
    else:
        for i in range(n * -1):
            result /= x
    return result


def power_func_recur(x, n):
    if n > 0:
        result = power_func_recur(x, n - 1) * x
    elif n < 0:
        result = power_func_recur(x, n + 1) / x
    else:
        return 1
    return result


print(power_func(x=4, n=-2))
print(power_func_recur(x=4, n=-2))


# Task 2


class Animal(object):
    def __init__(self):
        super().__init__()
        self._hunger_perc = 0.5

    def get_hunger_perc(self):
        return self._hunger_perc

    def eat(self):
        self._hunger_perc -= 0.1
        self._hunger_perc = max(0, self._hunger_perc)

    def move(self):
        pass

    def sleep(self, hours):
        self._hunger_perc += hours * 0.1
        self._hunger_perc = min(1, self._hunger_perc)


class Dog(Animal):
    def __init__(self):
        super().__init__()
        self.__bones_hidden = 0

    def move(self):
        self._hunger_perc += 1e-1
        self._hunger_perc = min(1, self._hunger_perc)

    def bark(self):
        print('bark')


class Cat(Animal):
    def __init__(self):
        super().__init__()
        self.__items_destroyed = 0

    def move(self):
        self._hunger_perc += 1e-2
        self._hunger_perc = min(1, self._hunger_perc)

    def meow(self):
        print('meow')


class Robot(object):
    def __init__(self):
        super().__init__()
        self.__battery_perc = 1.0

    def move(self):
        self.__battery_perc -= 1e-1
        self.__battery_perc = max(0, self.__battery_perc)

    def charge(self, hours):
        self.__battery_perc += 1e-1 * hours
        self.__battery_perc = min(1, self.__battery_perc)


dog_1 = Dog()
cat_1 = Cat()
robot_1 = Robot()
room_list = list()
room_list.append(dog_1)
room_list.append(cat_1)
room_list.append(robot_1)

for entity in room_list:
    entity.move()
    if isinstance(entity, Animal):
        entity.sleep(3)
        print(entity.get_hunger_perc())
        if isinstance(entity, Dog):
            entity.bark()
    elif isinstance(entity, Robot):
        entity.charge(3)


# Task 3


class Character(object):
    def __init__(self):
        super().__init__()
        self.geometry = []
        self.angle = 0.0
        self.speed = 0.1
        self.pos = np.array([0, 0])
        self.dir = np.array([0, 1])
        self.color = str('r')
        self.C = np.identity(3)
        self.R = np.identity(3)
        self.T = np.identity(3)

    def draw(self):
        x_data = []
        y_data = []
        for vec2 in self.geometry:
            x_data.append(vec2[0])
            y_data.append(vec2[1])
        plt.plot(x_data, y_data)

    def generate_geometry(self):
        pass


class Player(Character):
    def __init__(self):
        super().__init__()
        #self.polygon_data = [[0, self.pos[1] + 0.1], [self.pos[0] + 0.5, self.pos[1] - 0.5], [self.pos[0] - 0.5, self.pos[1] - 0.5]] POSSIBLE IMPLEMENTATION USING POLYGON

    def generate_geometry(self):
        self.geometry.append([self.pos[0], self.pos[1] + 0.1])
        self.geometry.append([self.pos[0] + 0.5, self.pos[1] - 0.5])
        self.geometry.append([self.pos[0] - 0.5, self.pos[1] - 0.5])
        self.geometry.append([self.pos[0], self.pos[1] + 0.1])

'''  def draw(self):
        x_data = []
        y_data = []
        for vec2 in self.geometry:
            x_data.append(vec2[0])
            y_data.append(vec2[1])
        plt.plot(x_data, y_data, color=self.color)
        #t1 = plt.Polygon(self.polygon_data, color=self.color, fill=False) POSSIBLE IMPLEMENTATION USING POLYGON
        #plt.gca().add_patch(t1) POSSIBLE IMPLEMENTATION USING POLYGON
'''


class Asteroid(Character):
    def __init__(self):
        super().__init__()

    def generate_geometry(self):
        self.geometry.append(self.pos)


characters = list()
player = Player()
player.generate_geometry()
characters.append(player)

characters.append(Asteroid())
characters.append(Asteroid())

is_running = True


def on_press(event):
    global is_running, player
    if event.key == 'escape':
        is_running = False
    elif event.key == 'left':
        player.angle += 5
    elif event.key == 'right':
        player.angle -= 5


fig, _ = plt.subplots()
fig.canvas.mpl_connect('key_press_event', on_press)

while is_running:
    plt.clf()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    for each in characters:
        each.draw()
        plt.title(f"Angle: {player.angle}")

    plt.draw()
    plt.pause(1e-3)