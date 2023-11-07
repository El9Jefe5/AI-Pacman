# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import gridworld

import random,util,math
import copy

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # Initialize a counter to store Q-values.
        self.qValues = util.Counter()
    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        # Check if the Q-value for the given state-action pair exists in the qValues dictionary.
        # If it does, return the corresponding Q-value; otherwise, return 0.0 as the default value.
        return self.qValues[(state, action)] if (state, action) in self.qValues else 0.0

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        # Initialize a list 'qvalues' to store the Q-values for all legal actions.
        # Check if there are legal actions (non-terminal state).
        # If there are legal actions, return the maximum Q-value among them.
        # If there are no legal actions (terminal state), return a value of 0.0.
        qvalues = [self.getQValue(state, action) for action in self.getLegalActions(state)]
        return max(qvalues) if len(qvalues) else 0.0

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        # Get a list of legal actions for the current state.
        legalActions = self.getLegalActions(state)

        # If there are no legal actions (terminal state), return None.
        if len(legalActions) == 0:
            return None

        # Calculate the value for the current state by calling computeValueFromQValues.
        value = self.computeValueFromQValues(state)

        # Create a list of actions that have the same Q-value as the calculated value.
        # These actions are considered equally valuable.
        actions = [
            action
            for action in legalActions
            if value == self.getQValue(state, action)
        ]
        ## Return a randomly chosen action from the list of equally valuable actions.
        return random.choice(actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        return (
            random.choice(legalActions)
            if util.flipCoin(self.epsilon)
            else self.computeActionFromQValues(state)
        )

    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        # Get the current Q-value for the given state-action pair.
        current_q_value = self.getQValue(state, action)

        # Calculate the updated Q-value using the Q-learning update rule.
        updated_q_value = (1 - self.alpha) * current_q_value + self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState))

        # Update the Q-value for the state-action pair in the Q-values dictionary.
        self.qValues[(state, action)] = updated_q_value
    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        # Get the current weight vector.
        weight = self.getWeights()

        # Obtain the feature vector for the given state-action pair using a feature extractor.
        featureVector = self.featExtractor.getFeatures(state,action)

        return sum(
            weight[feature] * featureValue
            for feature, featureValue in featureVector.items()
        )
    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        # Get the feature vector for the current state-action pair.
        featureVector = self.featExtractor.getFeatures(state, action)

        # Calculate the temporal difference (TD) error.
        # TD error is the difference between the predicted Q-value and the target Q-value.
        # It represents the error in our current Q-value estimation.
        diff = reward + self.discount * self.getValue(nextState) - self.getQValue(state, action)
        
        # Iterate through the features in the feature vector.
        for feature in featureVector:
          # Update the weight associated with the current feature.
          # This weight adjustment is based on the TD error, the learning rate (alpha),
          # and the value of the feature in the feature vector.
          # The weight update is part of the Q-learning algorithm with function approximation.
          self.weights[feature] += self.alpha * diff * featureVector[feature]
    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
