import numpy as np

def evolver(current):
    '''
    Function that evolves one step forward the state 'current' and returns it
    '''
    sidex = current.shape[0]
    sidey = current.shape[1]
    back = np.zeros(current.shape)

    for x in range(sidex):
        for y in range(sidey):
            neighbours = 0

            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    neighbours += current[(x+i+sidex)%sidex][(y+j+sidey)%sidey]
            
            neighbours -= current[x][y]

            if (current[x][y] == 1 and neighbours < 2):

                back[x][y] = 0

            elif(current[x][y] == 1 and neighbours > 3):

                back[x][y] = 0

            elif(current[x][y] == 0 and neighbours == 3):

                back[x][y] = 1

            else:

                back[x][y] = current[x][y]
    
    return back


# TO DO -> implement some check for periodic configurations
def activity(state, delta, check = False):
    '''
    Compute the 'activity' of a state, defined as activity=sum(abs(state1-state2))/(number of cells)
    '''
    a = state
    act = np.zeros(delta)
    for i in range(delta):
        b = evolver(a)
        act[i] = np.sum(np.abs(a-b))/(a.shape[1]*a.shape[0])
        if check:
            if act[i] == 0: # this means that the pattern is stable, either dead or not evolving, or evolving in particular ways
                if b.sum() == 0:
                    return 'dead'
                elif b.sum() != 0:
                    return 'stable'
        a = b
    if not check:
        return act
    elif check:
        return 'stable'

def create_set(grid_size, samples=10000, evolution=200):
    dataset = {'dead':[],'stable':[]}
    for i in range(samples):
        state = np.random.randint(0,2,grid_size)
        label = activity(state, evolution, check=True)
        dataset[label].append(state)
    return dataset
