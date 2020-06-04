import numpy as np

# Neural dictionary class for storing past observations alongside contexts, posteriors over kernels. Includes methods
# for appending new memories, computing similarity between contexts, and informing priors based on past experiences.

class NeuralDictionary():
    def __init__(self, hypothesis_space, generalization_rate, kernel_parameters):

        self.context_dict = {}
        self.hypothesis_space = hypothesis_space
        self.hypothesis_length = len(self.hypothesis_space)
        self.score_template = np.array([0. for i in range(self.hypothesis_length)])
        self.alpha = generalization_rate
        self.kernel_parameters = kernel_parameters
        self.penalties = np.exp(-0.15 * self.kernel_parameters)
        self.softmax_temp = 0.6

    def append(self, key, features, kernels, X, Y, action_values = None, uncertainty = None):
        self.context_dict[key] = [0, 0, 0, 0, 0, 0]
        self.context_dict[key][0] = features
        self.context_dict[key][1] = kernels
        self.context_dict[key][2] = X
        self.context_dict[key][3] = Y
        self.context_dict[key][4] = action_values
        self.context_dict[key][5] = uncertainty

    def cosine_similarity(self, vec_1, vec_2):
        self.numerator = sum(vec_1 * vec_2)
        self.length_1 = (sum(vec_1 * vec_1)) ** 0.5
        self.length_2 = (sum(vec_2 * vec_2)) ** 0.5
        self.denominator = self.length_1 * self.length_2
        self.dist = self.numerator / self.denominator
        return self.dist

    def softmax(self, array):
        self.probabilities = np.exp(array/self.softmax_temp) / (sum(np.exp(array/self.softmax_temp)))
        return self.probabilities

    def penalized_softmax(self, array):
        self.numerator = np.exp((array * self.penalties) / self.softmax_temp)
        self.denominator = np.sum(np.exp((array*self.penalties)/self.softmax_temp))
        self.probabilities = self.numerator / self.denominator
        return self.probabilities

    def look_up(self, current_context):
        self.scores = np.copy(self.score_template)

        for stored_context in self.context_dict:
            self.comparison_context = self.context_dict[stored_context][0]
            self.posterior = self.context_dict[stored_context][1]
            self.similarity = self.cosine_similarity(current_context, self.comparison_context)
            self.normalized_similarity = self.similarity*np.exp(-0.2*np.sum(self.comparison_context))

            self.scores += (self.normalized_similarity * self.posterior)

        return self.scores


    def create_informed_prior(self, prior, scores):
        self.untransformed_prior = prior + (self.alpha * scores)
        if scores.any() != 0.0:
            self.new_prior = self.penalized_softmax(self.untransformed_prior)
            #self.new_prior = self.softmax(self.untransformed_prior)
            return self.new_prior
        else:
            return self.untransformed_prior

    # Getter methods:
    def get_action_values(self, context):
        if context in self.context_dict:
            return self.context_dict[context][4]
        else:
            return False

    def get_action_confidence(self, context):
        if context in self.context_dict:
            return self.context_dict[context][5]
        else:
            return False