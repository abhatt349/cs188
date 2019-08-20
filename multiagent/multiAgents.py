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

    def evaluationFunction(self, currentGameState, action):
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
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newGhostPositions = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodDist = 0
        for food in newFood:
            foodDist += manhattanDistance(food, newPos)
        maxDist = 0
        for food1 in newFood:
            for food2 in newFood:
                dist = manhattanDistance(food1, food2)
                if dist > maxDist:
                    maxDist = dist
                    maxFood1 = food1
                    maxFood2 = food2
        if len(newFood) > 1:
            heurDist = maxDist + min(manhattanDistance(maxFood1, newPos), manhattanDistance(maxFood2, newPos))
        elif len(newFood) > 0:
            heurDist = manhattanDistance(newFood[0], newPos)
        else:
            heurDist = 0
        ghostDist = 0
        for ghost in newGhostPositions:
            ghostDist = manhattanDistance(ghost, newPos)
            if ghostDist == 0 or ghostDist == 1:
                return -500
        ghostDist = .5 / (.8 - ghostDist)
        #return 0 - foodDist - heurDist
        minDist = 10000
        for food in newFood:
            dist = manhattanDistance(newPos, food)
            if dist < minDist:
                minDist = dist
                minFood = food
        if len(currentGameState.getFood().asList()) > len(newFood):
            return 0
        if len(newFood) > 0:
            return 0 - minDist - (foodDist / (len(newFood) * 100))
        else:
            return 0




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

class Node:
    def __init__(self, gameState, agentIndex, actionList):
        self.children = []
        self.value = 0
        self.set = False
        self.state = gameState
        self.agentIndex = agentIndex
        self.actionList = actionList

    def populate(self, depth):
        if depth == 0:
            return
        for action in self.state.getLegalActions(self.agentIndex):
            newList = self.actionList + [action]
            self.children.append(Node(self.state.generateSuccessor(self.agentIndex, action), (self.agentIndex+1) % self.state.getNumAgents(), newList))
        for child in self.children:
            child.populate(depth - 1)

    def evaluate(self, evaluationFunction):
        if self.set:
            self.set = True
        elif self.state.isWin() or self.state.isLose() or (len(self.children) == 0):
            self.set = True
            self.value = evaluationFunction(self.state)
        elif self.agentIndex == 0:
            self.set = True
            self.value = max([child.evaluate(evaluationFunction) for child in self.children])
        else:
            self.set = True
            self.value = min([child.evaluate(evaluationFunction) for child in self.children])
        return self.value

    def getAction(self, depth, evaluationFunction):
        self.populate(depth)
        self.evaluate(evaluationFunction)

        if len(self.children) == 0:
            if len(self.state.getLegalActions(self.agentIndex)) == 0:
                return NULL
            else:
                return self.state.getLegalActions(agentIndex)[0]

        maxVal = self.children[0].value
        maxChild = self.children[0]
        for child in self.children:
            if child.value > maxVal:
                maxVal = child.value
                maxChild = child
        return maxChild.actionList[0]


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

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
        "*** YOUR CODE HERE ***"
        depth = self.depth * gameState.getNumAgents()
        head = Node(gameState, 0, [])
        return head.getAction(depth, self.evaluationFunction)

        util.raiseNotDefined()


class NodeAB:
    def __init__(self, gameState, agentIndex, actionList):
        self.children = []
        self.value = 0
        self.set = False
        self.state = gameState
        self.agentIndex = agentIndex
        self.actionList = actionList

    def evaluate(self, evaluationFunction, a, b, depth):
        if self.set:
            self.set = True
        elif self.state.isWin() or self.state.isLose() or depth == 0:
            self.set = True
            self.value = evaluationFunction(self.state)
        elif self.agentIndex == 0:
            self.set = True
            v = -999999
            for action in self.state.getLegalActions(self.agentIndex):
                newList = self.actionList + [action]
                child = NodeAB(self.state.generateSuccessor(self.agentIndex, action), (self.agentIndex+1) % self.state.getNumAgents(), newList)
                self.children.append(child)
                v = max(v, child.evaluate(evaluationFunction, a, b, depth-1))
                if v > b:
                    self.value = v
                    return self.value
                a = max(a, v)
                self.value = v
        else:
            self.set = True
            v = 999999
            for action in self.state.getLegalActions(self.agentIndex):
                newList = self.actionList + [action]
                child = NodeAB(self.state.generateSuccessor(self.agentIndex, action), (self.agentIndex+1) % self.state.getNumAgents(), newList)
                self.children.append(child)
                v = min(v, child.evaluate(evaluationFunction, a, b, depth-1))
                if v < a:
                    self.value = v
                    return self.value
                b = min(b, v)
                self.value = v
        return self.value


    def getAction(self, depth, evaluationFunction):
        self.evaluate(evaluationFunction, -999999, 999999, depth)

        if len(self.children) == 0:
            if len(self.state.getLegalActions(self.agentIndex)) == 0:
                return NULL
            else:
                return self.state.getLegalActions(agentIndex)[0]

        maxVal = self.children[0].value
        maxChild = self.children[0]
        for child in self.children:
            if child.value > maxVal:
                maxVal = child.value
                maxChild = child
        return maxChild.actionList[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        depth = self.depth * gameState.getNumAgents()
        head = NodeAB(gameState, 0, [])
        return head.getAction(depth, self.evaluationFunction)

        util.raiseNotDefined()


class NodeE:
    def __init__(self, gameState, agentIndex, actionList):
        self.children = []
        self.value = 0
        self.set = False
        self.state = gameState
        self.agentIndex = agentIndex
        self.actionList = actionList

    def evaluate(self, evaluationFunction, depth):
        if self.set:
            self.set = True
        elif self.state.isWin() or self.state.isLose() or depth == 0:
            self.set = True
            self.value = evaluationFunction(self.state)
        elif self.agentIndex == 0:
            self.set = True
            self.value = -999999
            for action in self.state.getLegalActions(self.agentIndex):
                newList = self.actionList + [action]
                child = NodeE(self.state.generateSuccessor(self.agentIndex, action), (self.agentIndex+1) % self.state.getNumAgents(), newList)
                self.children.append(child)
                self.value = max(self.value, child.evaluate(evaluationFunction, depth-1))
        else:
            self.set = True
            total = 0
            for action in self.state.getLegalActions(self.agentIndex):
                newList = self.actionList + [action]
                child = NodeE(self.state.generateSuccessor(self.agentIndex, action), (self.agentIndex+1) % self.state.getNumAgents(), newList)
                self.children.append(child)
                total += child.evaluate(evaluationFunction, depth-1)
            self.value = total / len(self.children)
        return self.value


    def getAction(self, depth, evaluationFunction):
        self.evaluate(evaluationFunction, depth)

        if len(self.children) == 0:
            if len(self.state.getLegalActions(self.agentIndex)) == 0:
                return NULL
            else:
                return self.state.getLegalActions(agentIndex)[0]

        maxVal = self.children[0].value
        maxChild = self.children[0]
        for child in self.children:
            if child.value > maxVal:
                maxVal = child.value
                maxChild = child
        return maxChild.actionList[0]



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        depth = self.depth * gameState.getNumAgents()
        head = NodeE(gameState, 0, [])
        return head.getAction(depth, self.evaluationFunction)

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacPos = currentGameState.getPacmanPosition()
    foodPos = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    ghostPositions = currentGameState.getGhostPositions()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    if currentGameState.isWin():
        return 100000
    if currentGameState.isLose():
        return -100000


    if len(foodPos) == 0:
        return 0

    minDist = 1000
    for food in foodPos:
        dist = manhattanDistance(food, pacPos)
        if dist < minDist:
            minDist = dist
            minFood = food
    return 0 - 15*len(foodPos) - minDist - 2*random.random()

    return 0 - len(foodPos)



    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction




















#
