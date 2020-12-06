# adversarialAgents.py
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
#
# Modified for use at University of Bath.


from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    """

    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation
        function.

        getAction takes a GameState and returns some Directions.X
        for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action)
                  for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores))
                       if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are
        better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class AdversarialSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    adversarial searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent and AlphaBetaPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(AdversarialSearchAgent):
    """
    Your minimax agent (question 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing
        minimax.

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

        def MaxValue(state, depth):
            if state.isWin() or state.isLose():  # end of game
                return state.getScore()

            value = float("-inf")  # negative infinity
            legal_moves = state.getLegalActions(0)
            best_move = Directions.STOP

            for move in legal_moves:
                new_value = MinValue(state.generateSuccessor(0, move), depth, 1)
                if new_value > value:
                    value = new_value
                    best_move = move

            if depth == 0:
                return best_move
            else:
                return value

        def MinValue(state, depth, ghost):
            if state.isWin() or state.isLose():  # end of game
                return state.getScore()

            value = float("inf")  # positive infinity
            legal_moves = state.getLegalActions(ghost)

            for move in legal_moves:
                if ghost == state.getNumAgents() - 1:  # last ghost
                    if depth == self.depth - 1:  # deepest point in search tree
                        new_value = self.evaluationFunction(state.generateSuccessor(ghost, move))
                    else:
                        new_value = MaxValue(state.generateSuccessor(ghost, move), depth + 1)
                else:
                    new_value = MinValue(state.generateSuccessor(ghost, move), depth, ghost + 1)
                if new_value < value:
                    value = new_value

            return value

        return MaxValue(gameState, 0)


class AlphaBetaAgent(AdversarialSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax with alpha-beta pruning action using self.depth and
        self.evaluationFunction
        """

        "*** YOUR CODE HERE ***"

        def MaxValue(state, depth, alpha, beta):
            if state.isWin() or state.isLose():  # end of game
                return state.getScore()

            value = float("-inf")  # negative infinity
            legal_moves = state.getLegalActions(0)
            best_move = Directions.STOP

            for move in legal_moves:
                new_value = MinValue(state.generateSuccessor(0, move), depth, 1, alpha, beta)
                if new_value > value:
                    value = new_value
                    best_move = move

                alpha = max(value, alpha)  # pruning
                if value > beta:
                    return value

            if depth == 0:
                return best_move
            else:
                return value

        def MinValue(state, depth, ghost, alpha, beta):
            if state.isWin() or state.isLose():  # end of game
                return state.getScore()

            value = float("inf")  # positive infinity
            legal_moves = state.getLegalActions(ghost)

            for move in legal_moves:
                if ghost == state.getNumAgents() - 1:  # last ghost
                    if depth == self.depth - 1:  # deepest point in search tree
                        new_value = self.evaluationFunction(state.generateSuccessor(ghost, move))
                    else:
                        new_value = MaxValue(state.generateSuccessor(ghost, move), depth + 1, alpha, beta)
                else:
                    new_value = MinValue(state.generateSuccessor(ghost, move), depth, ghost + 1, alpha, beta)
                if new_value < value:
                    value = new_value

                beta = min(beta, value)  # pruning
                if value < alpha:
                    return value

            return value

        return MaxValue(gameState, 0, float("-inf"), float("inf"))


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 3).

    DESCRIPTION: 
        The factors affecting the score are:
            a) Distance from ghosts
            b) Distance from food
            c) Win/lose
            d) Distance from capsules
        - Score increases when distance to food decreases, vice versa
        - Score decreases when distance to ghosts decreases, vice versa
        - Score affected greatly when win/lose.
        - Will use inverse distances to weight the scores. Distance from food/ghosts more important than distance to capsules.
    """
    score=0
    score += currentGameState.getScore()
    #Adds the game score to score total
    
    if currentGameState.isWin():
        #add points depending on win/loss
        score += 1000
    elif currentGameState.isLose():
        score -= 1000
    
    #finding key game data: pacman position, food positions, capsule positions, 
    #ghost states and the times ghosts are scared for
    currentPosition = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood().asList() 
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]  
    
    foodDistances=[]
    for food in currentFood:
        foodDistance = manhattanDistance(currentPosition, food)
        #for each food item, 2/distance from the pacman is added to an array if not already being eaten
        if(foodDistance>1):
            #food items weighted
            foodDistances.append(2/foodDistance)
    if(len(foodDistances)>0):
        #if 1+ food left, the optimal food item is added to score (closest food item)
        score += max(foodDistances)
    
    ghostDistances=[]
    for ghost in ghostStates:
        ghostDistance = manhattanDistance(currentPosition, ghost.getPosition())
        if(ghostDistance>1):
            ghostDistances.append(2/ghostDistance)
            #same process applied to ghosts as with food items.
            #added to array
    if(len(ghostDistances)>0):
        if(min(scaredTimes)>1):
            #if the ghosts are scared, it's optimal to go towards them and eat them
            score += max(ghostDistances)
        else:
            #if ghosts not scared, best to avoid the closest ones
            score -= max(ghostDistances)

    capsuleDistances=[]
    for capsule in capsules:
        capsuleDistance = manhattanDistance(currentPosition, capsule)
        capsuleDistances.append(1/capsuleDistance)
        #same process for identifying capsules. These are worth less so 1/distance is used as opposed to 2/distance.
    if(len(capsuleDistances)>0):
        #optimal to be closer to capsules
        score += max(capsuleDistances)
        
    return score
    #returns score

    util.raiseNotDefined()
