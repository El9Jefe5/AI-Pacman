# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Initialize the score with the current game score
        score = successorGameState.getScore()

        # Calculate the reciprocal of the closest food distance (to encourage approaching food)
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        closestFoodDistance = min(foodDistances) if foodDistances else 1  # Avoid division by zero
        score += 1.0 / closestFoodDistance

        # If ghosts are scared, avoid them; otherwise, consider the reciprocal of ghost distances
        for i, ghost in enumerate(newGhostStates):
            ghostPos = ghost.getPosition()
            scaredTime = newScaredTimes[i]
            ghostDistance = manhattanDistance(newPos, ghostPos)
        
        if scaredTime > 0:
            # If ghost is scared, approach it to eat it
            score += 10.0 / (ghostDistance + 1)  # High reward for approaching scared ghost
        elif ghostDistance <= 1:
            # If ghost is too close and not scared, penalize heavily to avoid it
            score -= 1000.0
        else:
            # If ghost is nearby but not too close, consider the reciprocal of its distance
            score += 1.0 / (ghostDistance + 1)

        return score

 

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        # Determining the number of ghosts in the game.
        GhostsNumber = gameState.getNumAgents() - 1

        # This is the maxLevel function, which represents the Pacman's turn.
        # It explores Pacman's actions to maximize the evaluation function.
        def maxLevel(gameState, depth):
            # Increase the depth for the current level.
            currentDepth = depth + 1

            # Check if the game is in a terminal state, or the depth limit is reached.
            if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
                # Return the evaluation of the current state.
                return self.evaluationFunction(gameState)

            # Initialize the maximum value to negative infinity.
            maxVal = float("-inf")

            # Get legal actions for Pacman (agentIndex 0).
            actions = gameState.getLegalActions(0)

            # Explore each action to find the maximum value.
            for action in actions:
                successor = gameState.generateSuccessor(0, action)
                # Update maxVal with the maximum value of minLevel.
                maxVal = max(maxVal, minLevel(successor, currentDepth, 1))

            # Return the maximum value found for Pacman.
            return maxVal

        # This is the minLevel function, which represents the ghosts' turn.
        # It explores ghost actions to minimize the evaluation function.
        def minLevel(gameState, depth, agentIndex):
            # Initialize the minimum value to positive infinity.
            minVal = float("inf")

            # Check if the game is in a terminal state.
            if gameState.isWin() or gameState.isLose():
                # Return the evaluation of the current state.
                return self.evaluationFunction(gameState)

            # Get legal actions for the current ghost (agentIndex).
            actions = gameState.getLegalActions(agentIndex)

            # Explore each action to find the minimum value.
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)

                if agentIndex == GhostsNumber:
                    # If the current agent is the last ghost, call maxLevel.
                    minVal = min(minVal, maxLevel(successor, depth))
                else:
                    # If there are more ghosts, call the next ghost's minLevel.
                    minVal = min(minVal, minLevel(successor, depth, agentIndex + 1))

            # Return the minimum value found for the current ghost.
            return minVal

        # Get legal actions for Pacman (agentIndex 0).
        actions = gameState.getLegalActions(0)

        # Initialize the current score to negative infinity.
        currScore = float("-inf")

        # Initialize the return action to an empty string.
        returnAction = ''

        # Iterate through Pacman's legal actions to find the best action.
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            # Calculate the score by calling minLevel for the first ghost.
            score = minLevel(nextState, 0, 1)

            # Check if the new score is better than the current score.
            if score > currScore:
                returnAction = action
                currScore = score

        # Return the action that leads to the highest score for Pacman.
        return returnAction
            

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        # This is the maxLevel function, which represents Pacman's turn. 
        # It explores Pacman's actions to maximize the evaluation function using alpha-beta pruning.
        def maxLevel(gameState, depth, alpha, beta):
            # Increase the depth for the current level.
            currentDepth = depth + 1

            # Check if the game is in a terminal state or if the depth limit is reached.
            if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
                # Return the evaluation of the current state.
                return self.evaluationFunction(gameState)

            # Initialize the maximum value to negative infinity.
            maxVal = float("-inf")

            # Iterate through legal actions for Pacman (agentIndex 0).
            for action in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, action)
                # Iterate through legal actions for Pacman (agentIndex 0).
                val = minLevel(successor, currentDepth, 1, alpha, beta)

                # Update maxVal with the maximum value found so far.
                maxVal = max(maxVal, val)
                # Update alpha to the maximum of alpha and maxVal.
                alpha = max(alpha, maxVal)
                
                # If maxVal is greater than or equal to beta, 
                # perform alpha-beta pruning and return maxVal.
                if maxVal > beta:
                    return maxVal

            # Return the maximum value found during Pacman's turn.
            return maxVal

        # This is the minLevel function, representing the ghosts' turn. 
        # It explores ghost actions to minimize the evaluation function using alpha-beta pruning. 
        def minLevel(gameState, depth, agentIndex, alpha, beta):
            # Check if the game is in a terminal state.
            if gameState.isWin() or gameState.isLose():
                # Return the evaluation of the current state.
                return self.evaluationFunction(gameState)

            # Initialize the minimum value to positive infinity.
            minVal = float("inf")

            # Iterate through legal actions for the current ghost (agentIndex).
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)

                if agentIndex == gameState.getNumAgents() - 1:
                    # If the current agent is the last ghost, 
                    # call maxLevel for the next level.
                    val = maxLevel(successor, depth, alpha, beta)

                else:
                    # If there are more ghosts, 
                    # call the next ghost's minLevel for the next level.
                    val =  minLevel(successor, depth, agentIndex + 1, alpha, beta)
                
                # Update minVal with the minimum value found so far.
                minVal = min(minVal, val)
                # Update beta to the minimum of beta and minVal.
                beta = min(beta, minVal)

                # If minVal is less than or equal to alpha, 
                # perform alpha-beta pruning and return minVal.    
                if minVal < alpha:
                        return minVal

            # Return the minimum value found during the ghosts' turn.    
            return minVal

        # Alpha-Beta Pruning
        # Initialize the current score to negative infinity.
        currScore = float("-inf")
        # Initialize the return action to an empty string.
        returnAction = ''
        # Initialize alpha and beta to negative infinity 
        # and positive infinity, respectively.
        alpha = float("-inf")
        beta = float("inf")

        # Iterate through legal actions for Pacman (agentIndex 0).
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0,action)
            # Calculate the score by calling minLevel for the first ghost.
            score = minLevel(nextState,0,1,alpha,beta)

            # Check if the new score is better than the current score.
            if score > currScore:
                returnAction = action
                currScore = score

            # Update alpha to the maximum of alpha and the new score.
            alpha = max(alpha,score)

        # Return the action that leads to the highest score for Pacman.
        return returnAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        # This is the maxLevel function, which represents Pacman's turn. 
        # It explores Pacman's actions to maximize the expected value using the expectLevel function.
        def maxLevel(gameState,depth):
            # Increase the depth for the current level.
            currentDepth = depth + 1

            # Check if the game is in a terminal state or if the depth limit is reached.
            if gameState.isWin() or gameState.isLose() or currentDepth==self.depth:   #Terminal Test 
                # Return the evaluation of the current state.
                return self.evaluationFunction(gameState)
            
            # Initialize the maximum value to negative infinity.
            maxVal = float("-inf")

            # Iterate through legal actions for Pacman (agentIndex 0).
            for action in gameState.getLegalActions(0):
                successor= gameState.generateSuccessor(0,action)
                # Recursively call expectLevel to get the expected value for this action.
                val = expectLevel(successor,currentDepth,1)

                # Update maxVal with the maximum expected value found so far.
                maxVal = max (maxVal, val)

            # Return the maximum expected value found during Pacman's turn.
            return maxVal
        
        # This is the expectLevel function, representing the ghosts' turn. 
        # It explores ghost actions to calculate the expected value.
        def expectLevel(gameState,depth, agentIndex):
            # Check if the game is in a terminal state.
            if gameState.isWin() or gameState.isLose():   #Terminal Test 
                # Return the evaluation of the current state.
                return self.evaluationFunction(gameState)
            
            # Initialize the total expected value to 0.
            totalExpectedVal = 0
            # Get the number of legal actions for the current agent.
            actionsNum = len(gameState.getLegalActions(agentIndex))

            # Get the number of legal actions for the current agent.
            for action in gameState.getLegalActions(agentIndex):
                successor= gameState.generateSuccessor(agentIndex,action)

                if agentIndex == (gameState.getNumAgents() - 1):
                    # If the current agent is the last ghost, 
                    # call maxLevel for the next level.
                    expectedVal = maxLevel(successor,depth)
                else:
                     # If there are more ghosts, 
                     # call the next ghost's expectLevel for the next level.
                    expectedVal = expectLevel(successor,depth,agentIndex+1)

                # Update the total expected value with the expected value for this action.
                totalExpectedVal = totalExpectedVal + expectedVal

            # If there are no legal actions, return 0 to avoid division by zero.
            if actionsNum == 0:
                return  0
            # Calculate the average expected value and return it.
            return float(totalExpectedVal)/float(actionsNum)
        
        # Initialize the current score to negative infinity.
        currScore = float("-inf")
        # Initialize the return action to an empty string.
        returnAction = ''

        # Iterate through legal actions for Pacman (agentIndex 0).
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0,action)
            # Calculate the score by calling expectLevel for the first ghost.
            score = expectLevel(nextState,0,1)

            # Calculate the score by calling expectLevel for the first ghost.
            if score > currScore:
                returnAction = action
                currScore = score
        
        # Return the action that leads to the highest expected value for Pacman.
        return returnAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: A sophisticated Pacman assessment feature that is excellent at ghost hunting, 
    food gobbling, and optimizing overall game performance.
    By taking into account a number of variables, this evaluation function is intended to assist 
    Pacman in making wise decisions.
    - It measures the distances between food pellets, incentivizing Pacman to aim for the closest ones.
    - It assists Pacman in avoiding risky situations by assessing the danger posed by ghosts and 
    their closeness.
    - To maximize point accumulation, it takes into account the current game score.
    - Pacman receives incentives from the function for attacking weak ghosts and consuming power pellets.
    - It also determines how many food and power pellets are left, preventing Pacman from wasting 
    important resources.

    This evaluation function's objective is to direct Pacman toward a successful strategy that aims to gather 
    as many points as possible while avoiding harmful ghosts and attacking them when they are at their 
    most vulnerable. In the Pacman game, this assessment function scores 1000 or higher on average, which 
    guarantees Pacman's victory in the majority of situations.

    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    from util import manhattanDistance
    
    # Get relevant information about game
    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    
    # Calculate distances to food and ghosts
    foodDistance = [manhattanDistance(pacmanPos, foodPos) for foodPos in foodList]
    ghostDistance = [manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in ghostStates]
    
    # Count the number of remaining food pellets, 
    # power pellets, and get the current game score
    numFood = len(foodList)
    numPowerPellets = len(currentGameState.getCapsules())
    score = currentGameState.getScore()
    
    # Initialize a list to store the score components
    scoreComponents = []
    
    # Add a score component based on the inverse 
    # of the distance to the closest food pellet
    if foodDistance:
        scoreComponents.append(1.0 / min(foodDistance))
    
    # Subtract a score component based on 
    # the distance to the closest ghost
    if ghostDistance:
        scoreComponents.append(-min(ghostDistance))
    
    # Add a score component based on the total scared times of ghosts
    scoreComponents.append(sum(scaredTimes))
    
    # Combine the score components by summing them up
    score += sum(scoreComponents)
    
    # Penalize for the number of power pellets 
    # and reward for the number of remaining food pellets
    score += numFood - numPowerPellets
    
    # Return the final score, 
    # which represents the evaluation of the game state
    return score

# Abbreviation
better = betterEvaluationFunction
