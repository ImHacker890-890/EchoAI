import numpy as np
import random
from collections import defaultdict

class LSTMCell:
    """Одна ячейка LSTM (упрощённая реализация)."""
    def __init__(self, input_size, hidden_size):
        # Веса для гейтов
        self.Wf = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.Wi = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.Wo = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.Wc = np.random.randn(input_size + hidden_size, hidden_size) * 0.01

    def forward(self, x, h_prev, c_prev):
        # Объединяем вход и предыдущее состояние
        concat = np.concatenate([x, h_prev])
        
        # Гейты
        forget_gate = self.sigmoid(np.dot(concat, self.Wf))
        input_gate = self.sigmoid(np.dot(concat, self.Wi))
        output_gate = self.sigmoid(np.dot(concat, self.Wo))
        cell_state = np.tanh(np.dot(concat, self.Wc))
        
        # Обновление состояния
        c_new = forget_gate * c_prev + input_gate * cell_state
        h_new = output_gate * np.tanh(c_new)
        
        return h_new, c_new

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
