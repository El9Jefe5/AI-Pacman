# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations):
            temp = util.Counter()  # Create a temporary counter to hold updated values.
            for state in self.mdp.getStates():
                # The value for a terminal state is set to 0.
                if self.mdp.isTerminal(state):
                    temp[state] = 0
                else:
                    maximumValue = float("-inf")  # Initialize the maximum value to negative infinity.
                    actions = self.mdp.getPossibleActions(state)  # Get a list of possible actions for the state.
                    for action in actions:
                        t = self.mdp.getTransitionStatesAndProbs(state, action)  # Get transition states and probabilities.
                        # Calculate the Q-value for the current action in the state.
                        value = sum(
                            stateAndProb[1] * (
                                self.mdp.getReward(state, action, stateAndProb[1])  # Immediate reward
                                + self.discount * self.values[stateAndProb[0]]  # Discounted future value
                            )
                            for stateAndProb in t  # Iterate over possible successor states.
                        )
                        maximumValue = max(value, maximumValue)  # Update the maximum value.
                    if maximumValue != float("-inf"):
                        temp[state] = maximumValue  # Update the value for the state.
            self.values = temp  # Update the values with the temporary counter.


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # The Q-value for the given action in the state is calculated as a weighted sum of future rewards.
    # It sums over all possible successor states of the given state under the specified action.
        return sum(
            stateAndProb[1]
            * (
                self.mdp.getReward(state, action, stateAndProb[1]) # Immediate reward
                + self.discount * self.values[stateAndProb[0]] # Discounted future value
            )
            for stateAndProb in self.mdp.getTransitionStatesAndProbs(state, action) # Iterate over possible successor states.
        )

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        # Check if the state is terminal; if so, return None since no actions are possible at the terminal state.
        if self.mdp.isTerminal(state):
            return None

        # Get a list of legal actions for the current state.
        actions = self.mdp.getPossibleActions(state)

        # Initialize a dictionary 'allActions' to store the computed Q-values for each action in the current state.
        allActions = {
            action: self.computeQValueFromValues(state, action)
            for action in actions
        }

        # Return the action that maximizes the Q-value, breaking ties arbitrarily if there are multiple actions with the same Q-value.
        return max(allActions, key=allActions.get)

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
