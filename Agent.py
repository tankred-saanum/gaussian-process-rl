import numpy as np

# A class whose methods are useful for computing action values based on differing criteria, and generating actions
# informed by these values, either stochastically (softmax) or deterministically (maximizing value)

class Agent():
    def __init__(self, input_space, beta, temperature):
        self.action_space = input_space
        self.beta = beta
        self.temperature = temperature
        self.actions_taken = np.array([1 for i in range(len(input_space))])



    def computeUCB(self, predicted_values):

        self.total_choices = np.sum(self.actions_taken)
        self.uncertainty = np.sqrt(np.log(self.total_choices) / self.actions_taken)
        self.values = predicted_values
        self.ucb = self.values + (self.beta * self.uncertainty)
        return self.ucb

    def gpUpperConfidence(self, predicted_values, gp_variance):

        self.uncertainty = np.sqrt(gp_variance)
        self.gp_ucb = predicted_values + (self.beta * self.uncertainty)
        return self.gp_ucb

    def computeSoftmax(self, values):

        self.numerator = np.exp(values/self.temperature)
        self.denominator = np.sum(self.numerator)
        self.choice_probabilities = self.numerator/self.denominator
        return self.choice_probabilities

    def stochastic_action(self, option_values, actions):

        self.probabilities = self.computeSoftmax(option_values)
        self.action = np.random.choice(actions, p=self.probabilities)
        return self.action

    def deterministic_action(self, option_values, actions):
        self.action_index = np.argmax(option_values)
        self.action = actions[self.action_index]
        self.actions_taken[self.action_index] += 1
        return self.action




