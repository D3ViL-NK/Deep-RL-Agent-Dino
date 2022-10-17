# Imports
import tensorflow as tf
import os
from keras.optimizers import Adam
from keras import layers
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from collections import deque
import random
import numpy as np
from PIL import ImageGrab
import cv2
import time
import pytesseract
from win32 import win32gui
from pynput.keyboard import Key
from pynput.keyboard import Controller as key_Controller

#Keyboard controling Object
keyboard = key_Controller()


# DQN_Agent Parameter assignment
batch_size = 32
n_episodes = 100000
output_dir = 'Models'

#Agent Class
class DQNAgent:

    def __init__(self, state_size, action_size):
        
        self.state_size = state_size    #Input data size
        self.action_size = action_size  #How many action that the bot is able to take
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):

        
        # 2D convolutional Model
        
        # model = Sequential()
        # model.add(layers.Conv2D(32, (3,3), activation ='relu', input_shape=(13,50,1)))
        # model.add(layers.MaxPooling2D(2,2))
        # model.add(layers.Conv2D(64, (3,3), activation ='relu'))
        # model.add(layers.MaxPooling2D((2,2)))
        # model.add(layers.Conv2D(64,(3,3), activation='relu'))
        # model.summary()
        # model.add(layers.Flatten())
        # model.add(layers.Dense(64, activation = 'relu'))
        # model.add(Dense(self.action_size, activation = 'relu'))

        # All connected nural Network
        model = Sequential()
        model.add(layers.Flatten(input_dim=self.state_size))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(self.action_size, activation='relu'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.summary()
        print(self.state_size)
        return model

    # Storing the last state action reward and the next state in memory
    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    # Predicting the next action
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        done == False
        for (state, action, reward, next_state) in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    # Loading a saved model weights and biases
    def load(self, name):
        self.model = tf.keras.models.load_model(name)
    # Saving a trained model
    def save(self, name):
        self.model.save(name)

##IMAGE CAPTURE AND GAME RESET###

# Preproccessing the receved image data and before feeding the bot

def process_img(original_img):
    process_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    process_img = cv2.Canny(process_img, threshold1=200, threshold2=300)
    return process_img


# names = win32gui.GetWindowText(win32gui.GetForegroundWindow()) # Getting opend window names

# Finding the window and getting the dimenstions
hwnd = win32gui.FindWindow(None, r'chrome://dino/ - Google Chrome')
win32gui.SetForegroundWindow(hwnd)
dimensions = win32gui.GetWindowRect(hwnd)
x1, y1, x2, y2 = dimensions

state_size = 50*13 #Setting the state_size to the size of the image that feeding to the model.
action_size = 3

agent = DQNAgent(state_size, action_size) #Defining the agent object

for e in range(n_episodes):  # Running the agent for n episodes

    #Resting the values for the new episode
    reward = 0
    done = False
    
    while(done == False):
        
        im_L = ImageGrab.grab((x1+20, y1+250, x2*2.8/3, y2-(y2)*7.17/12))# Captureing the window to compare
        # (x1+20, y1+250, x2*2.8/3, y2-(y2)*7.17/12) doing some mathematical manipulations to the dimentions to remove the frame of the window.
        
        next_state = []
        state = []
        action = 0

        image = ImageGrab.grab((x1+20, y1+250, x2*2.8/3, y2-(y2)*7.17/12))
        
        if im_L == image: #Checking for game over situation.
            
            image = image.resize((50, 13))
            screen = np.array(image)
            next_state = process_img(screen)
            # print(state.shape)
            next_state = np.reshape(next_state, [1, state_size])
            rew = -100
            
            if not(state == []):
                agent.remember(state, action, rew, next_state)
            
            # print("Game_over")
            
            keyboard.press(Key.space)
            cv2.waitKey(650)
            keyboard.release(Key.space)
            print("episode: {}/{}, training_score:{}, e: {:.2}".format(e,
                                                                       n_episodes, reward, agent.epsilon))
            done = True
            break
            
        if done == True: # Ending the episode if the game is over.
            break
            
        reward += 1 #Positive reward for evry time it stays alive
        rew = 1 # Current reward
        image = image.resize((50, 13))
        screen = np.array(image)
        state = process_img(screen)
        temp_state = state
        state = np.reshape(state, [1, state_size])
        action = agent.act(state)
        agent.remember(state, action, rew, next_state)
        if action == 1:
            keyboard.press(Key.space)
            keyboard.release(Key.space)
        elif action == 2:
            keyboard.press(Key.down)
            keyboard.release(Key.down)

        # print("FPS: {}".format(1/(time.time()-time1)))
        # time1 = time.time()
        
        next_state = state
        
        # display the current state

        # cv2.imshow('Current State',cv2.cvtColor(temp_state,cv2.COLOR_BGR2RGB))
        # if cv2.waitKey(25) & 0xtemp_state == ord('q'):
        #     cv2.destroyAllWindows()
        #     break

    # Retraning the nural network with new memory
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    # Saving the model at every 50 episodes
    if e % 50 == 0:
        agent.save(output_dir + "/model_" + '{:04d}'.format(e) + ".h5")

