import random
import math
import csv

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def derivative_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

class NeuralNetwork:
    def __init__(self):
        self.w1 = random.normalvariate(0, 1)
        self.w2 = random.normalvariate(0, 1)
        self.b = random.normalvariate(0, 1)

    def feedforward(self, x):
        return sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b)

    def train(self, data, y_trues):
        learn_rate = 0.1
        epochs = 1000 

        for epoch in range(epochs):
            for x, y_true in zip(data, y_trues):
                sum_h = self.w1 * x[0] + self.w2 * x[1] + self.b
                h = sigmoid(sum_h)

                d_L_d_ypred = -2 * (y_true - h)
                
                d_ypred_d_w1 = x[0] * derivative_sigmoid(sum_h)
                d_ypred_d_w2 = x[1] * derivative_sigmoid(sum_h)
                d_ypred_d_b = derivative_sigmoid(sum_h)

                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_w2
                self.b -= learn_rate * d_L_d_ypred * d_ypred_d_b

# Load data from CSV file
data = []
y_trues = []
user_ids = {}
message_ids = {}
next_user_id = 0
next_message_id = 0
with open('chat_data.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header row
    for row in reader:
        if row[0] not in user_ids:
            user_ids[row[0]] = next_user_id
            next_user_id += 1
        if row[1] not in message_ids:
            message_ids[row[1]] = next_message_id
            next_message_id += 1
        data.append([user_ids[row[0]], message_ids[row[1]]])
        y_trues.append(float(row[2]))

network = NeuralNetwork()
network.train(data, y_trues)

# Chatbot loop
while True:
    message = input("Enter a message: ")
    if message in message_ids:
        message_id = message_ids[message]
        response = network.feedforward([0, message_id])  # Assuming user ID is 0
        print("Response: " + str(response))
    else:
        print("I'm not sure how to respond to that. Let me try...")
        # Generate a response using the neural network
        response = network.feedforward([0, next_message_id])  # Assuming user ID is 0
        print("Response: " + str(response))
        
        print("What would you suggest I say?")
        suggested_response = input()
        message_ids[message] = next_message_id
        next_message_id += 1
        data.append([0, message_ids[message]])  # Assuming user ID is 0
        y_trues.append(float(suggested_response))
        network.train(data, y_trues)
