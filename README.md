# RLDino
### Objective
Making a Deep Reinforcement Learning agent to play the Game Dino.
![Game](https://github.com/D3ViL-NK/RLDino/blob/main/Game.png)


### Main roles of the system
* Capturing the window.
* Resizing, preprocessing and feeding the image(current state) to the agent.
* Predicting the action using the Reinforcemnt learning agent.
* Depending on the received action giving keybord input to the game.
* Using next state assign a reward for the agent.

### Reinforcement Learning agent
#### Methods
* _build_model: Making new model.
* load: Loading a trained model's weights and biases.
* save: Saving the model's weights and biases.
* remember: Saving the current state, action, reward and the next state.
* replay: Retraining the nuralnet using the recent experience.
* act: Deciding next action.(predicting the next best action using the predicted reward or take a random action.)
