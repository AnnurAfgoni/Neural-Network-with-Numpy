class Neuron:
    
    def __init__(self, input_size, learning_rate, epoch):
        """
            input_size = jumlah baris dari input
        """
        self.w = np.random.rand(input_size, 1)
        self.bias = np.random.rand(1, 1)
        self.learning_rate = learning_rate
        self.epoch = epoch
        
    # method for feedforward
    def f_forward(self, inp):
        """
            inp = data input
        """
        result = inp.T.dot(self.w) + self.bias
        return result
    
    # mean squared error
    def loss(self, x, y):
        """
            x = data input
            y = real output
        """
        a = self.f_forward(x)
        L = (a - y)**2
        return L
    
    # backpropagation
    def backward(self, x, y):
        """
            x = data input
            y = real output
        """
        a = self.f_forward(x)
        
        dw = 2.*(a - y) * x.T
        db = 2.*(a - y) * 1.
        
        # gradient descent
        self.w = self.w - self.learning_rate * dw.T
        self.bias = self.bias - self.learning_rate * db.T
        
        return self.w, self.bias
    
    def train(self, x, y):
        """
            x = data input
            y = real output
        """
        acc = []
        losss = []
        for j in range(self.epoch):
            l = []
            for i in range(x.shape[0]):
                l = self.loss(x, y)
                w1, b1 = self.backward(x, y)
            print("epochs:", j + 1, "======== acc:", (1 - (sum(l)/x.shape[0]))*100)
            acc.append((1 - (sum(l)/x.shape[0]))*100)
            losss.append(sum(l)/x.shape[0])
        return (acc, losss, w1, b1)
    
    """def prediksi(self, x, y, value):
        pass """