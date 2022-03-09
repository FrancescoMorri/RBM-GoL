from matplotlib import rc
rc('animation', html='jshtml')
import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 2**128
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from helper_functions import evolver


class Animate():
    def __init__(self, starting_state):
        self.state = starting_state
    
    def gen(self):
        i = 0
        while self.state.sum() != 0 and i < 100:
            i += 1
            yield i
    
    
    
    def plot_anim(self):
        fig,ax = plt.subplots()

        def animate(i):
                ax.imshow(self.state, cmap='binary')
                ax.set_title("STEP %d"%(i))
                self.state = evolver(self.state)
        
        anim = animation.FuncAnimation(fig, animate, frames=self.gen, repeat=False, interval=100)
        return anim
