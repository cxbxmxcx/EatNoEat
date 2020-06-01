import tensorflow as tf
from collections import deque
from tensorflow import keras
import numpy as np
import random

EPISODES = 500
TRAIN_END = 0

#Hyper Parameters
def discount_rate(): #Gamma
    return 0.95

def learning_rate(): #Alpha
    return 0.001

def batch_size(): #Size of the batch used in the experience replay
    return 24

class DeepQNetwork():
  def __init__(self, states, actions, alpha, gamma, epsilon,epsilon_min, epsilon_decay):
    self.nS = states
    self.nA = actions
    self.memory = deque([], maxlen=2500)
    self.alpha = alpha
    self.gamma = gamma
    #Explore/Exploit
    self.epsilon = epsilon
    self.epsilon_min = epsilon_min
    self.epsilon_decay = epsilon_decay
    self.model = self.build_model()
    self.loss = []
        
  def build_model(self):
    model = keras.Sequential() #linear stack of layers https://keras.io/models/sequential/
    model.add(keras.layers.Dense(24, input_dim=self.nS, activation='relu')) #[Input] -> Layer 1
    #   Dense: Densely connected layer https://keras.io/layers/core/
    #   24: Number of neurons
    #   input_dim: Number of input variables
    #   activation: Rectified Linear Unit (relu) ranges >= 0
    model.add(keras.layers.Dense(24, activation='relu')) #Layer 2 -> 3
    model.add(keras.layers.Dense(self.nA, activation='linear')) #Layer 3 -> [output]
    #   Size has to match the output (different actions)
    #   Linear activation on the last layer
    model.compile(loss='mean_squared_error', #Loss function: Mean Squared Error
                  optimizer=keras.optimizers.Adam(lr=self.alpha)) #Optimaizer: Adam (Feel free to check other options)
    return model

  def action(self, state):
    if np.random.rand() <= self.epsilon:
        return random.randrange(self.nA) #Explore
    action_vals = self.model.predict(state) #Exploit: Use the NN to predict the correct action from this state
    return np.argmax(action_vals[0])

  def test_action(self, state): #Exploit
    action_vals = self.model.predict(state)
    return np.argmax(action_vals[0])

  def store(self, state, action, reward, nstate, done):
    #Store the experience in memory
    self.memory.append( (state, action, reward, nstate, done) )

  def experience_replay(self, batch_size, episode):
    #Execute the experience replay
    minibatch = random.sample( self.memory, batch_size ) #Randomly sample from memory

    #Convert to numpy for speed by vectorization
    x = []
    y = []
    np_array = np.array(minibatch)
    st = np.ones((0,self.nS)) #States
    nst = np.ones( (0,self.nS) )#Next States
    for i in range(len(np_array)): #Creating the state and next state np arrays
        st = np.append( st, np_array[i,0], axis=0)
        nst = np.append( nst, np_array[i,3], axis=0)
    st_predict = self.model.predict(st) #Here is the speedup! I can predict on the ENTIRE batch
    nst_predict = self.model.predict(nst)
    index = 0
    for state, action, reward, nstate, done in minibatch:
        x.append(state)
        #Predict from state
        nst_action_predict_model = nst_predict[index]
        if done == True: #Terminal: Just assign reward much like {* (not done) - QB[state][action]}
            target = reward
        else:   #Non terminal
            target = reward + self.gamma * np.amax(nst_action_predict_model)
        target_f = st_predict[index]
        target_f[action] = target
        y.append(target_f)
        index += 1
    #Reshape for Keras Fit
    x_reshape = np.array(x).reshape(batch_size,self.nS)
    y_reshape = np.array(y)
    epoch_count = random.randint(1,5) #Epochs is the number or iterations
    hist = self.model.fit(x_reshape, y_reshape, epochs=epoch_count, verbose=0)
    #Graph Losses
    with tf.name_scope('Training'):
      tf.summary.scalar('losses', data=max(hist.history['loss']), step=episode) 
    for i in range(epoch_count):
        self.loss.append( hist.history['loss'][i] )
    #Decay Epsilon
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay

def train_dqn(training_folder):
    #Create the agent
    nS = 3
    nA = 2
    dqn = DeepQNetwork(nS, nA, learning_rate(), discount_rate(), 1, 0.001, 0.9985 )

    batch_size = batch_size()

    test_foods = pickle.load( open( training_folder + "food_test.p", "rb" ) )
    print(test_foods)

    foods = len(test_foods)
    tests_per_food = 30
    next_food = random.choice(test_foods)
    tot_rewards = 0
    for ft in nb.tqdm(range(tests_per_food * foods)):  
      food = next_food
      state = food[0]  
      action = dqn.action(state)
      reward = 1 if food[1] == action else -1  
      tot_rewards += reward
      next_food = random.choice(test_foods)
      nstate = next_food[0]
      done = False
      dqn.store(state, action, reward, nstate, done) # Resize to store in memory to pass to .predict
      with tf.name_scope('Training'):
        tf.summary.scalar('rewards', data=tot_rewards, step=ft) 
        tf.summary.scalar('epsilon', data=dqn.epsilon, step=ft)  
      if len(dqn.memory) > batch_size:
          dqn.experience_replay(batch_size, ft)
  