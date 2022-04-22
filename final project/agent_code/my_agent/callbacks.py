import os
import pickle
import random
import torch
import io
import numpy as np
from random import shuffle
from .Qnet import Q_network,device,train_step, give_action

#ACTIONS = ['LEFT','RIGHT','UP', 'DOWN','WAIT','BOMB']
ACTIONS = ['LEFT','RIGHT','UP', 'DOWN','WAIT','BOMB']

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        #weights = np.random.rand(len(ACTIONS))
        #self.model = weights / weights.sum()
        self.model =  Q_network().to(device)
        self.model_t = Q_network().to(device)
        self.model_t.load_state_dict(self.model.state_dict())
        print(self.model)
    else:
        self.logger.info("Loading model from saved state.")
        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                else: return super().find_class(module, name)
        with open("my-saved-model.pt", "rb") as file:
            #device = torch.device("cpu")
            self.model = CPU_Unpickler(file).load()
            print(self.model)
            
    self.steps = 0
    self.testStep = 50


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    #print("steps:" +str(self.steps))
    self.steps += 1
    random_prob = .95
    self.channel = state_to_features( game_state)
    #print(game_state)
    if self.train:
        if self.steps < 5000:
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .15, 0.05])  
            
            
        if game_state['round']  % self.testStep == 0:
            
            #self.logger.debug("Querying model for action.")
            ind = give_action(self.model, self.channel)
            avatar = game_state['self'][3]
            coins  = game_state['coins']
            #print(game_state['field'])
            #print(game_state['self'])
            #print(game_state['coins'])
            #print("nearest:")
            #print(nearest_coin(avatar, coins))
            #print(self.channel)
            #print("call-act:"+str(game_state['self'][1]) + "next act: "+ACTIONS[ind])
            return ACTIONS[ind]
        
        if random.random() < random_prob:
            # todo Exploration vs exploitation
            #self.logger.debug("Choosing action purely at random.")
            # 80%: walk in any direction. 10% wait. 10% bomb.
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .15, 0.05])
    
    #print(game_state['self'][1])      
    ind = give_action(self.model, self.channel)
    #avatar = game_state['self'][3]
    #coins  = game_state['coins']
    #print(game_state['field'])
    #print(game_state['self'])
    #print(game_state['coins'])
    #print("nearest:")
    #print(nearest_coin(avatar, coins))
    #print(self.channel)
    #print("call-act:"+str(game_state['self'][1]) + "next act: "+ACTIONS[ind])
    
    #self.logger.debug("Querying model for action."+str(ACTIONS[ind])+" socre: " + str(game_state["self"][1]))
    return ACTIONS[ind]


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """ 
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    features = []

    field = game_state['field']
    avatar = game_state['self'][3]
    x,y = avatar
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(field.shape) * 5
    free_space = field == 0
    
    # While this timer is positive, agent will not hunt/attack opponents
    ignore_others_timer = 0 
    if ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    cols = range(1, field.shape[0] - 1)
    rows = range(1, field.shape[0] - 1)
    
    dead_ends = [(x, y) for x in cols for y in rows if (field[x, y] == 0)
                 and ([field[x + 1, y], field[x - 1, y], field[x, y + 1], field[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in cols for y in rows if (field[x, y] == 1)]
    targets = coins + dead_ends + crates
    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    

    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)
                
    ###############1 check if the left of the avatar is wall, if true append 1, else append 0
    #check wall around
    wall_channel = [-1,-1,-1,-1] #left, right, up, down 1-free, -1 -wall
    left = (x-1,y)
    if (field[left] == 0 and
                (game_state['explosion_map'][left] < 1) and
                (bomb_map[left] > 0) and
                (not left in others) and
                (not left in bomb_xys)):
           wall_channel[0] = 1
    right = (x+1,y)     
    if (field[right] == 0 and
                (game_state['explosion_map'][right] < 1) and
                (bomb_map[right] > 0) and
                (not right in others) and
                (not right in bomb_xys)):
           wall_channel[1] = 1
     
    up = (x, y-1)
    d  = up      
    if (field[d] == 0 and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
           wall_channel[2] = 1
    down = (x, y+1)
    d = down       
    if (field[d] == 0 and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
           wall_channel[3] = 1   

    #################2 check the direction of the nearest coins
    #coin direction channel: #left, right, up, down  yes 1, no 0
    dir_channel = [0,0,0,0] 
    target = look_for_targets(free_space, (x, y), targets, None)
    if target == (x, y - 1): dir_channel = [0,0,1,0]
    if target == (x, y + 1): dir_channel = [0,0,0,1]
    if target == (x - 1, y): dir_channel = [1,0,0,0]
    if target == (x + 1, y): dir_channel = [0,1,0,0]

    
    
    #################3 dangerous path? [0,0,0,0] no dangerous, [1,0,0,0] left is dangerous
    #[0,1,0,0] - right is dangerous, [0,0,1,0], up is dangerous, [0,0,0,1] down is dangerous 
    path =  agent_bombs_samepath(avatar, bombs, field)

     #print(game_state)
    #################4 Touch crate? index 12
    crate = touch_crate(avatar, field)
    
    #################5 Gived bomb ? index 13
    bombed = [1] #agent can release one bomb
    if game_state['self'][2]:
        bombed = [0]   #agent can not give bomb now
    
    #################6 bomb around
    bombaround = [0] #0-no bomb, 1 - have bomb
    if bomb_around(avatar, bombs):
        bombaround = [1] 
        
    #################7 touch coin
    tcoin = [0]
    if len(coins) != 0:
        tcoin = touch_coin(avatar, coins)
        
    ##################8 dead ends
    fdead_ends = [0]
    if avatar in dead_ends:
        fdead_ends = [1]
         
    
    #features += wall_channel + dir_channel + path + crate + bombed + bombaround + tcoin
    features += wall_channel + dir_channel + path + crate + bombed + bombaround + tcoin + fdead_ends
    #print(features)
    return np.array(features)
def touch_coin(agent,  coins):
    x,y = agent
    for coin in coins:
        if coin in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
            return [1]
    return [0]
def touch_crate(agent, field):
    x,y = agent
    if [field[x + 1, y], field[x - 1, y], field[x, y + 1], field[x, y - 1]].count(1) > 0:
        return [1]
    return [0]
    
#find the nearest coin    
def nearest_coin(avatar, coins, field):
    #print(field)
    #start from the avatar to find the nearest coin based on bfs
    visited = [avatar]
    queue   = [avatar]
    distance_so_far = {avatar:0}
    parent_dict = {avatar:avatar}
    best = avatar
    #not_found = True
    while(len(queue) != 0):
        obj = queue.pop(0)
        x1,y1 = obj
        neighbours = [(x, y) for (x, y) in [(x1 + 1, y1), (x1 - 1, y1), (x1, y1 + 1), (x1, y1 - 1)] if (field[y, x] != -1 and (x,y) not in visited)]
        for (x,y) in neighbours:
            parent_dict[(x,y)] = (x1,y1)
            distance_so_far[(x,y)] = distance_so_far[(x1,y1)] + 1
            if ((x,y) in coins) or (field[(x,y)] == 1):
                best = (x,y)
                queue = []
                #not_found = False
                break;     
            else:
                queue.append((x,y))
                visited.append((x,y))
                
    if best == avatar:
        print("not found")
        return 100, [0,0,0,0]
    current = best
    #print("goal",best)
    while parent_dict[current] != avatar:
        current = parent_dict[current]
    feature = [0,0,0,0]
    #print("first step", current)
    if current == (avatar[0]-1, avatar[1]):
        feature = [1,0,0,0]
    elif current == (avatar[0]+1, avatar[1]):
        feature = [0,1,0,0]
    elif current == (avatar[0], avatar[1]-1):
        feature = [0,0,1,0]
    elif current == (avatar[0], avatar[1]+1):
        feature = [0,0,0,1]
    return distance_so_far[best], feature

def dead_ends(field):
    cols = range(1, field.shape[0] - 1)
    rows = range(1, field.shape[0] - 1)
    dead_ends = [(x, y) for x in cols for y in rows if (field[x, y] == 0) and ([field[x + 1, y], field[x - 1, y], field[x, y + 1], field[x, y - 1]].count(0) == 1)]
    if dead_ends:
        return [1]
    else:
        return [0]
def bomb_around(avatar, bombs):
    x,y = avatar
    for bomb in bombs:
        bx,by = bomb[0]
        if abs(x-bx) + abs(y-by) < 4:
            return True
            
    return False
        
def agent_bombs_samepath(avatar, bombs, field):
    if len(bombs) == 0:
        return [0,0,0,0]
    x,y = avatar
    for bomb in bombs:
        bomb = bomb[0]
        bx,by = bomb
        if avatar == bomb:
            return [1,1,1,1]
        if x == bx and abs(y - by) < 4:
            y2 = max(y,by)
            y1 = min(y,by)
            if np.sum(field[x,y1:y2] == -1) == 0:
                if by > y:
                    return [0,0,0,1] #the forth pos(D): 0 no bottom, 1 bottom
                else:
                    return [0,0,1,0] #the third pos(u): 0 no bottom, 1 bottom
                
        if y == by and abs(x - bx) < 4:
            x2 = max(x,bx)
            x1 = min(x,bx)
            if np.sum(field[x1:x2,y] == -1) == 0:
                if bx < x:
                    return [1,0,0,0]
                else:
                    return [0,1,0,0]              
    return [0,0,0,0]
            
    
def free(pos, field):
    if field[pos] == 0:
        return True
    else:
        return False
        
        
def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]
