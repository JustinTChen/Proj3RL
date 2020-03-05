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
        i, temp = 0, util.Counter()
        while i < self.iterations:
            for state in self.mdp.getStates():
                
                V_list = []
                if self.mdp.isTerminal(state):
                    V_list.append(0)
                for action in self.mdp.getPossibleActions(state):
                    V_list.append(self.computeQValueFromValues(state, action))
                
                temp[state] = max(V_list)
            
            self.values = temp.copy()
            i += 1

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
        
        V_list = []
        for transition in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, transition[0])
            value_next = self.values[transition[0]]
            V_list.append(transition[1] * (reward + self.discount * value_next))
        return sum(V_list)

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        best_value, take_action = -1 * float("inf"), None         
        for action in self.mdp.getPossibleActions(state):
            curr_value = self.computeQValueFromValues(state, action)
            if curr_value > best_value:
                best_value = curr_value
                take_action = action
        return take_action

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
        
        self.values = util.Counter()
        states = self.mdp.getStates()
        
        for i in range(0, self.iterations):
            state = states[i % len(states)]
            if self.mdp.isTerminal(state):
                continue
            action = self.getAction(state)
            self.values[state] = self.computeQValueFromValues(state, action)


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
        #"""
        states = self.mdp.getStates()
        
        #First, map nodes to their predecessors
        pred = dict()
        for state in states:
            #print(state)
            if self.mdp.isTerminal(state):
                continue
            for action in self.mdp.getPossibleActions(state):
                for t_state, p in self.mdp.getTransitionStatesAndProbs(state, action):
                    try:
                        pred[t_state].add(state)
                    except:
                        pred[t_state] = {state}
        #print(pred)
        
        #Initialize Priority Queue
        PQ = util.PriorityQueue()
        
        #Diffs
        for state in states:
            if not self.mdp.isTerminal(state):
                
                values = []
                best = -1 * float("inf")
                for action in self.mdp.getPossibleActions(state):
                    q_value = self.computeQValueFromValues(state, action)
                    values.append(q_value)
                    best = max(best, q_value)
                    
                diff = abs(self.values[state] - best)
                PQ.update(state, -1 * diff)
        
    
        #Iterate
        i = 0
        while i < self.iterations and not PQ.isEmpty():
            state = PQ.pop()
            self.values[state] = (max([self.computeQValueFromValues(state, action) 
                                    for action in self.mdp.getPossibleActions(state)]))
            
            for p_state in pred[state]:
                if self.mdp.isTerminal(p_state):
                    continue
                
                q_values = ([self.computeQValueFromValues(p_state, action) 
                            for action in self.mdp.getPossibleActions(p_state)])
    
                diff = abs(self.values[p_state] - max(q_values))
                
                if diff > self.theta:
                    PQ.update(p_state, -1 * diff)
                    
            i += 1