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



class Actions:
    """
    A collection of static methods for manipulating move actions.
    """
    # Directions
    _directions = {Directions.NORTH: (0, 1),
                   Directions.SOUTH: (0, -1),
                   Directions.EAST:  (1, 0),
                   Directions.WEST:  (-1, 0),
                   Directions.STOP:  (0, 0)}

    _directionsAsList = _directions.items()

    TOLERANCE = .001

    def reverseDirection(action):
        if action == Directions.NORTH:
            return Directions.SOUTH
        if action == Directions.SOUTH:
            return Directions.NORTH
        if action == Directions.EAST:
            return Directions.WEST
        if action == Directions.WEST:
            return Directions.EAST
        return action
    reverseDirection = staticmethod(reverseDirection)

    def vectorToDirection(vector):
        dx, dy = vector
        if dy > 0:
            return Directions.NORTH
        if dy < 0:
            return Directions.SOUTH
        if dx < 0:
            return Directions.WEST
        if dx > 0:
            return Directions.EAST
        return Directions.STOP
    vectorToDirection = staticmethod(vectorToDirection)

    def directionToVector(direction, speed = 1.0):
        dx, dy =  Actions._directions[direction]
        return (dx * speed, dy * speed)
    directionToVector = staticmethod(directionToVector)

    def getPossibleActions(config, walls):
        possible = []
        x, y = config.pos
        x_int, y_int = int(x + 0.5), int(y + 0.5)

        # In between grid points, all agents must continue straight
        if (abs(x - x_int) + abs(y - y_int)  > Actions.TOLERANCE):
            return [config.getDirection()]

        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_y = y_int + dy
            next_x = x_int + dx
            if not walls[next_x][next_y]: possible.append(dir)

        return possible

    getPossibleActions = staticmethod(getPossibleActions)

    def getLegalNeighbors(position, walls):
        x,y = position
        x_int, y_int = int(x + 0.5), int(y + 0.5)
        neighbors = []
        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_x = x_int + dx
            if next_x < 0 or next_x == walls.width: continue
            next_y = y_int + dy
            if next_y < 0 or next_y == walls.height: continue
            if not walls[next_x][next_y]: neighbors.append((next_x, next_y))
        return neighbors
    getLegalNeighbors = staticmethod(getLegalNeighbors)

    def getSuccessor(position, action):
        dx, dy = Actions.directionToVector(action)
        x, y = position
        return (x + dx, y + dy)
    getSuccessor = staticmethod(getSuccessor)




class PositionSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost




def nodedCost(parent, state, direction, cost):
    #ancestors = parent[0].append(state)
    #dirList = parent[1].append(direction)
    ancestors = []
    for thing in parent[0]:
        ancestors.append(thing)
    dirList = []
    for thing in parent[1]:
        dirList.append(thing)

    ancestors.append(state)
    dirList.append(direction)

    return (ancestors, dirList, parent[2] + cost)


def aStarSearch(problem, heuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    closed = set([])
    fringe = util.PriorityQueue()
    fringe.push(([problem.getStartState()], [], 0), heuristic(problem.getStartState(), problem))
    while (not fringe.isEmpty()):
        node = fringe.pop()
        if (problem.isGoalState(node[0][-1])):
            return node[1]
        if (not (node[0][-1] in closed)):
            closed.add(node[0][-1])
            for (child, dir, cost) in problem.getSuccessors(node[0][-1]):
                childNode = nodedCost(node, child, dir, cost)
                fringe.push(childNode, childNode[2] + heuristic(child, problem))


    util.raiseNotDefined()

def manhattanHeuristic(position, problem):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def mazeDistance(point1, point2, gameState, dict = {}):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    if (point1, point2) in dict.keys():
        return dict[(point1, point2)]
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    dict[(point1, point2)] = len(aStarSearch(prob, manhattanHeuristic))
    return dict[(point1, point2)]



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
        return (10 * currentGameState.getScore())
    if currentGameState.isLose():
        return -100000


    if len(foodPos) == 0:
        print("bruh")
        return 0

    minDist = 1000
    for food in foodPos:
        dist = mazeDistance(food, pacPos, currentGameState)
        if dist < minDist:
            minDist = dist
            minFood = food
    #return 0 - 15*len(foodPos) - minDist - 2*random.random()

    if len(foodPos) == 1:
        return 0 - mazeDistance(foodPos[0], pacPos, currentGameState)

    maxDist = 0
    for food1 in foodPos:
        for food2 in foodPos:
            dist = mazeDistance(food1, food2, currentGameState)
            if dist > maxDist:
                maxDist = dist
                maxFood1 = food1
                maxFood2 = food2
    dist = min(mazeDistance(pacPos, maxFood1, currentGameState), mazeDistance(pacPos, maxFood2, currentGameState)) + maxDist
    return 0 - 10*len(foodPos) - 2*dist - minDist - random.random()

    return 0 - len(foodPos)



    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction




















#
