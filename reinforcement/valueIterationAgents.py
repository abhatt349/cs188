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
    def __init__(self, mdp, discount = 0.9, iterations = 100):
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
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        n = self.iterations
        while(n != 0):
            mdp = self.mdp
            discount = self.discount
            copy = self.values.copy()
            values = self.values
            for state in self.mdp.getStates():
                actions = mdp.getPossibleActions(state)
                if len(actions) == 0:
                    values[state] = 0
                else:
                    values[state] = max(sum(prob * (mdp.getReward(state, action, nextState) + (discount * copy[nextState])) for (nextState, prob) in mdp.getTransitionStatesAndProbs(state, action)) for action in actions)
            n -= 1

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
        mdp = self.mdp
        discount = self.discount
        values = self.values
        return sum(prob * (mdp.getReward(state, action, nextState) + (discount * values[nextState])) for (nextState, prob) in mdp.getTransitionStatesAndProbs(state, action))
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        ctr = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            ctr[action] = self.computeQValueFromValues(state, action)
        return ctr.argMax()
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        n = self.iterations
        while(n != 0):
            mdp = self.mdp
            discount = self.discount
            values = self.values
            for state in self.mdp.getStates():
                if n == 0:
                    return
                n -= 1
                actions = mdp.getPossibleActions(state)
                if len(actions) == 0:
                    values[state] = 0
                else:
                    values[state] = max(sum(prob * (mdp.getReward(state, action, nextState) + (discount * values[nextState])) for (nextState, prob) in mdp.getTransitionStatesAndProbs(state, action)) for action in actions)



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        values = self.values
        discount = self.discount
        predecessors = {}
        for state in mdp.getStates():
            preList = []
            for preState in mdp.getStates():
                for action in mdp.getPossibleActions(preState):
                    if state in [pair[0] for pair in mdp.getTransitionStatesAndProbs(preState, action) if pair[1] > 0]:
                        preList.append(preState)
                        break
            predecessors[state] = preList
        queue = util.PriorityQueue()
        for s in mdp.getStates():
            if not mdp.isTerminal(s):
                actions = mdp.getPossibleActions(s)
                realValue = max(sum(prob * (mdp.getReward(s, action, nextState) + (discount * values[nextState])) for (nextState, prob) in mdp.getTransitionStatesAndProbs(s, action)) for action in actions)
                diff = abs(realValue - values[s])
                queue.push(s, 0 - diff)
        for _ in range(self.iterations):
            if queue.isEmpty():
                return
            s = queue.pop()
            if not mdp.isTerminal(s):
                actions = mdp.getPossibleActions(s)
                values[s] = max(sum(prob * (mdp.getReward(s, action, nextState) + (discount * values[nextState])) for (nextState, prob) in mdp.getTransitionStatesAndProbs(s, action)) for action in actions)
            for p in predecessors[s]:
                actions = mdp.getPossibleActions(p)
                realValue = max(sum(prob * (mdp.getReward(p, action, nextState) + (discount * values[nextState])) for (nextState, prob) in mdp.getTransitionStatesAndProbs(p, action)) for action in actions)
                diff = abs(realValue - values[p])
                if diff > self.theta:
                    queue.update(p, 0 - diff)





















#
