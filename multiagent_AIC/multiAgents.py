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
import math

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP
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

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman_AIC.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman_AIC.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostsPos = [ghostState.getPosition() for ghostState in newGhostStates]
        
        # YOUR CODE
        foodList = newFood.asList()
        foodDistances = [manhattanDistance(newPos, food) for food in foodList]
        ghostDistances = [manhattanDistance(newPos, ghost) for ghost in newGhostsPos]

        # closest food and ghost
        closestFood = min(foodDistances) if len(foodDistances) > 0 else 0
        closestGhost = min(ghostDistances) if len(ghostDistances) > 0 else 0

        # score calculation for food and ghost
        closestFoodScore = 0 if closestFood == 0 else 1.0/closestFood
        closestGhostScore = 0 if closestGhost == 0 else 1.0/closestGhost

        return successorGameState.getScore() + closestFoodScore - closestGhostScore

def scoreEvaluationFunction(currentGameState):
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
    Your minimax agent (question 5)
    """
   
    # maximaxer agent (pacman) function
    def MAX_VALUE(self, gameState, d):
        if d == 0 or gameState.isWin() or gameState.isLose():
            # base case: return evaluation function
            return self.evaluationFunction(gameState), Directions.STOP
        
        bestScore, bestAction = -math.inf, Directions.STOP

        for action in gameState.getLegalActions(0):
            successors = gameState.generateSuccessor(0, action)

            # call minimaxer agent (ghosts)
            value, _ = self.MIN_VALUE(successors, d, 1)

            # update best score and action
            if value > bestScore:
                bestScore, bestAction = value, action

        return bestScore, bestAction

    # minimizer agent (ghosts) function
    def MIN_VALUE(self, gameState, d, indexAgent):
        if d == 0 or gameState.isWin() or gameState.isLose():
            # base case: return evaluation function
            return self.evaluationFunction(gameState), Directions.STOP

        bestScore, bestAction = math.inf, Directions.STOP

        for action in gameState.getLegalActions(indexAgent):
            successors = gameState.generateSuccessor(indexAgent, action)
            if indexAgent == gameState.getNumAgents() - 1:
                # call pacman
                value, _ = self.MAX_VALUE(successors, d - 1)
            else:
                # call next ghost
                value, _ = self.MIN_VALUE(successors, d, indexAgent + 1)

            # update best score and action
            if value < bestScore:
                bestScore, bestAction = value, action

        return bestScore, bestAction
   
    def getAction(self, gameState):
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
        
        evaluation, action = self.MAX_VALUE(gameState, self.depth)

        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 6)
    """
    
    def MAX_VALUE(self, gameState, d, alpha, beta):
        if d == 0 or gameState.isWin() or gameState.isLose():
            # base case: return evaluation function
            return self.evaluationFunction(gameState), Directions.STOP
        
        bestScore, bestAction = -math.inf, Directions.STOP

        for action in gameState.getLegalActions(0):
            successors = gameState.generateSuccessor(0, action)

            # call minimaxer agent (ghosts)
            value, _ = self.MIN_VALUE(successors, d, 1, alpha, beta)

            if value > beta:
                return value, action

            alpha = max(alpha, value)

            # update best score and action
            if value > bestScore:
                bestScore, bestAction = value, action

        return bestScore, bestAction

    # minimizer agent (ghosts) function
    def MIN_VALUE(self, gameState, d, indexAgent, alpha, beta):
        if d == 0 or gameState.isWin() or gameState.isLose():
            # base case: return evaluation function
            return self.evaluationFunction(gameState), Directions.STOP

        bestScore, bestAction = math.inf, Directions.STOP

        for action in gameState.getLegalActions(indexAgent):
            successors = gameState.generateSuccessor(indexAgent, action)
            if indexAgent == gameState.getNumAgents() - 1:
                # call pacman
                value, _ = self.MAX_VALUE(successors, d - 1, alpha, beta)
            else:
                # call next ghost
                value, _ = self.MIN_VALUE(successors, d, indexAgent + 1, alpha, beta)

            if value < alpha:
                return value, action

            beta = min(beta, value)

            # update best score and action
            if value < bestScore:
                bestScore, bestAction = value, action

        return bestScore, bestAction

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        evaluation, action = self.MAX_VALUE(gameState, self.depth, -9999, 9999)

        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 7)
    """

    def MAX_VALUE(self, gameState, d, alpha, beta):
        if d == 0 or gameState.isWin() or gameState.isLose():
            # base case: return evaluation function
            return self.evaluationFunction(gameState), Directions.STOP
        
        bestScore, bestAction = -math.inf, Directions.STOP

        for action in gameState.getLegalActions(0):
            successors = gameState.generateSuccessor(0, action)

            # call minimaxer agent (ghosts)
            value, _ = self.MIN_VALUE(successors, d, 1, alpha, beta)

            if value > beta:
                return value, action

            alpha = max(alpha, value)

            # update best score and action
            if value > bestScore:
                bestScore, bestAction = value, action

        return bestScore, bestAction

    # minimizer agent (ghosts) function
    def MIN_VALUE(self, gameState, d, indexAgent, alpha, beta):
        if d == 0 or gameState.isWin() or gameState.isLose():
            # base case: return evaluation function
            return self.evaluationFunction(gameState), Directions.STOP

        bestScore, bestAction = 0, Directions.STOP

        for action in gameState.getLegalActions(indexAgent):
            successors = gameState.generateSuccessor(indexAgent, action)
            if indexAgent == gameState.getNumAgents() - 1:
                # call pacman
                value, _ = self.MAX_VALUE(successors, d - 1, alpha, beta)
            else:
                # call next ghost
                value, _ = self.MIN_VALUE(successors, d, indexAgent + 1, alpha, beta)

            if value < alpha:
                return value, action

            beta = min(beta, value)
            
            value = value / len(gameState.getLegalActions(indexAgent))
            bestScore += value
            bestAction = action


            # # update best score and action
            # if value < bestScore:
            #     bestScore, bestAction = value, action

        return bestScore, bestAction

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
            
        evaluation, action = self.MAX_VALUE(gameState, self.depth, -9999, 9999)

        return action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 8).

    DESCRIPTION: <write something here so we know what you did>
    """
    
    def closestFoodDistance(gameState):
        pacmanPosition = gameState.getPacmanPosition()
        foodList = gameState.getFood().asList()

        if len(foodList) == 0:
            return 0

        return min([manhattanDistance(pacmanPosition, food) for food in foodList])

    closestFood = closestFoodDistance(currentGameState)
    return currentGameState.getScore() - (10 * closestFood) - (100 * currentGameState.getNumFood())

# Abbreviation
better = betterEvaluationFunction