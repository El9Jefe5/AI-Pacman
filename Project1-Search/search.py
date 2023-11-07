# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""


from typing import Deque
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    hasVisited = set()  # This allows us too keep track of visited states
    stack = []  # Stack for DFS traversal essentially for the frontier
    startingState = problem.getStartState()
    actions = []

    # Every item in our stack is a tuple (state, actions)
    stack.append((startingState, actions))

    #Creating a loop that keeps running as long as there are elements in the stack 
    #Meaning the condition will evaluate to true
    while stack:
        #Here we retrieve the current state and path taken to get to it
        currentState, actions = stack.pop()
        #If our current state is the goal state
        if problem.isGoalState(currentState):
            #We return the path taken to get to the goal
            return actions

        #If our current state has not been visited yet
        if currentState not in hasVisited:
            #Let's add it to our set and mark it as visited
            hasVisited.add(currentState)
            #Here we get the states that come after our current state; its successors
            successors = problem.getSuccessors(currentState)
            #Loop to iterate through the elements of successors ignoring the third element of the tuple
            for nextState, action, unused in successors:
                #add the currrent path to the one we already have
                newActions = actions + [action]
                #Here we add the state of our successor and newly acquired path to the stack
                stack.append((nextState, newActions))

    return actions  # If there is no solution, we simply return an empty list of paths or rroutes taken


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    hasVisited = set()  # This set allows us to keep track of visited states
    bfsQueue = Deque() # breadthFirstSearch queue
    startingState = problem.getStartState()
    actions =[]

    # Every item in the queue is a tuple representing (state, actions)
    bfsQueue.append((startingState, actions))

    while bfsQueue:
        #Here we retrieve the current state and path taken to get to it 
        #And use popleft() instead of pop(0) to remove the first element from the left of the queue since we're doing BFS
        #This for me was only a matter of preference. Conceptually popleft() represents better the idea of bfs at least for me
        currentState, actions = bfsQueue.popleft()
        #If our current state is the goal state
        if problem.isGoalState(currentState):
            #We return the path taken to get to the goal
            return actions

        #If our current state has not been visited yet
        if currentState not in hasVisited:
            #Let's add it to our set and mark it as visited
            hasVisited.add(currentState)
             #Here we get the states that come after our current state; its successors
            successors = problem.getSuccessors(currentState)
            #Loop to iterate through the elements of successors ignoring the third element of the tuple
            for nextState, action, unused in successors:
                #add the currrent path to the one we already have
                newActions = actions + [action]
                #Here we add the state of our successor and newly acquired path to the stack
                bfsQueue.append((nextState, newActions))

    return actions  # If there is no solution, we simply return an empty list of paths or routes taken


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # We Initialize a priority queue with the start state and a cost of 0.
    ucsQueue = util.PriorityQueue()
    startingState = problem.getStartState()
    actions = []
    ucsQueue.push((startingState, actions, 0), 0)

    # Then Keep track of visited states.
    hasVisited = set()
    #While our queue is not empty
    while not ucsQueue.isEmpty():
        #We retrieve the current state, the path taken to get to it, and the cost
        state, actions, cost = ucsQueue.pop()

        #if the state is a goal state
        if problem.isGoalState(state):
            #We simply return the list of actions or path taken to get there
            return actions

        #If the current state has not yet been visited 
        if state not in hasVisited:
            #Add it to our set and mark it up as visted now since we just did
            hasVisited.add(state)
            #get the states that come after our current state
            successors = problem.getSuccessors(state)
            #Loop to iterate through the elements of successors this time not ignoring the third value of the tuple
            for nextState, action, stepCost in successors:
                #add the currrent path to the one we already have
                newActions = actions + [action]
                #We then compute a new cost where we add the previous cost that we had to the new cost of the successor
                totalCost = cost + stepCost
                #Here we add the state of our successor, newly acquired path and new cost to the priority queue
                ucsQueue.push((nextState, newActions, totalCost), totalCost)

    return actions  # If there is no solution, we simply return an empty list of paths or routes taken as required

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Initialize a priority queue with the start state, an empty list of actions, and a cost of 0.
    aStarQueue = util.PriorityQueue()
    #Current state or beginning state
    currentStartState = problem.getStartState()
    actions = []
    aStarQueue.push((currentStartState, actions, 0), 0)

    # Keep track of visited states, doing so by creating a set, maintaining and updating it
    hasVisited = set()

    #When our priority queue is not empty
    while not aStarQueue.isEmpty():
        #We retrieve our current state, the path taken to get to it and the cost
        state, actions, cost = aStarQueue.pop()

        #If the current state is a goal state
        if problem.isGoalState(state):
            #Give us the path that's gotten us there
            return actions

        #If the current state has not yet been visited 
        if state not in hasVisited:
            #add it to our set marking it as visited
            hasVisited.add(state)
            #get the next state or successor
            successors = problem.getSuccessors(state)
            #Loop to iterate through the elements of successors this time not ignoring the third value of the tuple
            for nextState, action, stepCost in successors:
                #add the currrent path to the one we already have
                newActions = actions + [action]
                #We then compute a new cost where we add the previous cost that we had to the new cost of the successor
                costNew = cost + stepCost
                costHeuristic = heuristic(nextState, problem)  # Calculate the heuristic cost h(n). The Heuristic itself takes two args: state and search problem
                totalCost = costNew + costHeuristic  # Total cost is g(n) + h(n). This is where we add the heuristic
                aStarQueue.push((nextState, newActions, costNew), totalCost)

    return actions  # If there is no solution, we simply return an empty list of paths or routes taken as required



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
