class Attention:
    """Простой механизм внимания для анализа контекста."""
    def __init__(self, hidden_size):
        self.W = np.random.randn(hidden_size, hidden_size) * 0.01

    def forward(self, hidden_states):
        # Вычисляем веса внимания
        scores = np.dot(hidden_states, self.W)
        attention_weights = self.softmax(scores)
        
        # Взвешенная сумма
        context = np.sum(hidden_states * attention_weights, axis=0)
        return context

    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=0)
