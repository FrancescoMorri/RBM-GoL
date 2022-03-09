from time import time
import numpy as np
import torch
from tqdm import tqdm


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


def autocorr_calc(x, lags):
    corr=[1. if l==0 else np.corrcoef(x[l:],x[:-l])[0][1] for l in lags]
    return np.array(corr)

def check_const(func):
    check = []
    for h in range(len(func)):
        for k in range(h,len(func)):
            check.append(func[h]-func[k])

    return np.all((np.array(check) == 0))

def activity(state, delta, check = False, termalizing=50):
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
        if check_const(act[termalizing:]):
            return 'stable'
        autocorr = autocorr_calc(act[int(termalizing):], np.arange(0, int(len(act[int(termalizing):])/2)))
        if 1 in autocorr[1:]:
            return 'periodic'
        else:
            return 'stable'


def create_set(grid_size, samples, evolution=200, equity=False):
    dataset = {'dead':[],'stable':[],'periodic':[]}
    if not equity:
        for i in range(samples):
            state = np.random.randint(0,2,grid_size)
            label = activity(state, evolution, check=True, termalizing=50)
            dataset[label].append(state)
        return dataset
    else:
        start_time = time()
        count_dead = 0
        time_dead = True
        count_stable = 0
        time_stable = True
        count_periodic = 0
        time_periodic = True
        loop_flag = True
        while loop_flag:
            state = np.random.randint(0,2,grid_size)
            label = activity(state, evolution, check=True, termalizing=50)
            if label == 'dead' and count_dead < samples:
                count_dead += 1
                dataset['dead'].append(state)
            
            elif label == 'stable' and count_stable < samples:
                count_stable += 1
                dataset['stable'].append(state)
            
            elif label == 'periodic' and count_periodic < samples:
                count_periodic += 1
                dataset['periodic'].append(state)
            
            if count_periodic == samples and count_dead == samples and count_stable == samples:
                loop_flag = False
            
            if count_periodic == samples and time_periodic:
                print("Finished sampling periodic solution")
                print("Elapsed time: %.3f minutes"%((time()-start_time)/60))
                time_periodic = False
            
            if count_dead == samples and time_dead:
                print("Finished sampling dead solution")
                print("Elapsed time: %.3f minutes"%((time()-start_time)/60))
                time_dead = False
            
            if count_stable == samples and time_stable:
                print("Finished sampling stable solution")
                print("Elapsed time: %.3f minutes"%((time()-start_time)/60))
                time_stable = False

        
        return dataset


def training(model, data_loader, epochs, optimizer, schedule_lr=False, scheduler=False):
    tot_loss = []
    energy = []
    for epoch in tqdm(range(epochs)):
        loss_ = []
        for i, (data, target) in enumerate(data_loader):
            v, v1 = model(data)
            loss = model.free_energy(v) - model.free_energy(v1)
            loss_.append(loss.data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss.append(np.mean(loss_))
    return tot_loss


def testing(model, data_loader, sizex, sizey):
    real_data = []
    rbm_data = []
    for data, _ in tqdm(data_loader):
        v, v1 = model(data)
        v = torch.squeeze(v)
        tmp = v.view(sizex, sizey)
        real_data.append(tmp.data)
        v1 = torch.squeeze(v1)
        tmp = v1.view(sizex, sizey)
        rbm_data.append(tmp.data)
    return real_data, rbm_data