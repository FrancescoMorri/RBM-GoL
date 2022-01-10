import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from helper_functions import evolver

fig, ax = plt.subplots()

starting_state = np.random.randint(0,2,(4,4))

def gen():
    global starting_state
    i = 0
    while starting_state.sum() != 0:
        i += 1
        yield i

def animate(i):
    global starting_state
    ax.imshow(starting_state, cmap='binary')
    ax.set_title("STEP %d"%(i))
    starting_state = evolver(starting_state)


anim = animation.FuncAnimation(fig, animate, frames=gen, repeat=False)

plt.show()