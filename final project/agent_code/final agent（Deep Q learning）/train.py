from collections import namedtuple, deque

import pickle
from typing import List
from .Qnet import train_step
import events as e
from .callbacks import state_to_features
# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3000000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
NEARCOIN_EVENT = 'NERSESTCOIN'
AWAYCOIN_EVENT = 'AWAYCOIN'
AWAYBOMB_EVENT = 'AWAYBOMB'
AWAYBOMB_EVENT1 = 'AWAYBOMB1'
GOOD_BOMB = 'GOODBOMB'
BAD_BOMB = 'BADBOMB'
GOOD_WAIT = 'GOODWAIT'
TOUCH_EVENT = 'TOUCH'
CLOSE2BOMB_EVENT = 'CLOSE2BOMB'
ACTIONS = ['LEFT','RIGHT','UP', 'DOWN','WAIT','BOMB']
def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.channel = None
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.sum = 0
    self.count_ = 1


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    #self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    if(self.steps % 4 == 0):
        return
    

    # Idea: Add your own events to hand out rewards
    feature  = state_to_features(old_game_state)
    #feature1 = state_to_features(new_game_state)        
    ld = feature[8]         #if left side has bomb
    rd = feature[9]         #if down side has..
    ud = feature[10]        #if up side ..
    dd = feature[11]        #if down side..
    bomb_dropped = feature[13]
    touch_crate  = feature[12]
    touch_coin   = feature[15]
    bombaround   = feature[14]
    dead_ends    = feature[16]
    if touch_coin == 1 or touch_crate == 1:
        events.append(TOUCH_EVENT)
    
    if 'INVALID_ACTION' not in events:

        if ACTIONS.index(self_action) < 4 and feature[4:8].sum() != 0:
            #print(feature[4:8])
            if feature[ACTIONS.index(self_action)+4] == 1:
                events.append(NEARCOIN_EVENT)
            else:
                events.append(AWAYCOIN_EVENT)
        
        
         #run away from bomb
        if ACTIONS.index(self_action) < 4:
            index = ACTIONS.index(self_action) 
            if (feature[8:12].sum() == 4):
                events.append(AWAYBOMB_EVENT1)
            else:
                if(ld == 1):
                    if (index == 2 or index == 3):
                        events.append(AWAYBOMB_EVENT)
                    elif(index == 1):
                        events.append(AWAYBOMB_EVENT1)
                    else:
                        events.append(CLOSE2BOMB_EVENT)
                if(rd == 1):
                    if (index == 2 or index == 3):
                        events.append(AWAYBOMB_EVENT)
                    elif(index == 0):
                        events.append(AWAYBOMB_EVENT1)
                    else:
                        events.append(CLOSE2BOMB_EVENT)
                        
                if(ud == 1):
                    if (index == 0 or index == 1):
                        events.append(AWAYBOMB_EVENT)
                    elif(index == 3):
                        events.append(AWAYBOMB_EVENT1)
                    else:
                        events.append(CLOSE2BOMB_EVENT)
                        
                if(dd == 1):
                    if (index == 0 or index == 1):
                        events.append(AWAYBOMB_EVENT)
                    elif(index == 2):
                        events.append(AWAYBOMB_EVENT1)
                    else:
                        events.append(CLOSE2BOMB_EVENT)
                
              
        #drop bom
        if (self_action == 'BOMB'  and  touch_crate == 0) or (touch_crate == 1 and self_action != 'BOMB' and bomb_dropped != 1):
            events.append(BAD_BOMB)
              
        if self_action == 'BOMB' and (touch_crate  == 1 or dead_ends==1):
            events.append(GOOD_BOMB)
            
        if self_action == 'WAIT' and  bombaround == 1  and (feature[8:12].sum() == 0): #current position is safe
            events.append(GOOD_WAIT)
            
        
    
        #print("now:" + str(new_game_state['self'][1]) )
    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), ACTIONS.index(self_action), state_to_features(new_game_state), reward_from_events(self, events)))
    #print(reward_from_events(self, events))
    self.sum += reward_from_events(self, events)
    if new_game_state['round'] % self.testStep == 0 and self.steps > 5000:
        self.logger.debug("scores: "+str(new_game_state['self'][1]))
        print(events)
        print(self.sum)
    if self.steps > 5000:
        train_step(self.model, self.model_t, self.transitions)
    #print("trainsitions:"+str(len(self.transitions)))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    if last_game_state['round'] % self.testStep == 0 and self.steps > 5000:
        self.logger.debug("scores: "+str(last_game_state['self'][1]))
        print("last:" + str(last_game_state['self'][1]) )
        print("rewards:"+str(self.sum))
    self.sum = 0   
    #update the parameters of target_network
    if last_game_state['round'] % 5 == 0:
        self.model_t.load_state_dict(self.model.state_dict())
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    
    self.transitions.append(Transition(state_to_features(last_game_state), ACTIONS.index(last_action), None, reward_from_events(self, events)))
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 2,
        e.KILLED_OPPONENT: 5,
        e.INVALID_ACTION:-0.5,
        e.KILLED_SELF:-5,
        e.WAITED:-0.05,
        e.CRATE_DESTROYED:0.8,
        e.COIN_FOUND:0.8,
        BAD_BOMB:-0.2,
        GOOD_BOMB:0.5,
        TOUCH_EVENT:0.05,
        GOOD_WAIT:0.1,
        NEARCOIN_EVENT:0.2,
        AWAYCOIN_EVENT:-0.25,
        AWAYBOMB_EVENT:0.4,
        AWAYBOMB_EVENT1:0.1,
        CLOSE2BOMB_EVENT:-0.9,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
    
   
   


