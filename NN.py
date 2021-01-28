from math import exp
import abc
import random
import matplotlib.pyplot as plt


# class for weight with value and the input node
class Weight:
    def __init__(self,w):
        self.w = w
        self.loss = 0
    def calculate_loss(self):
        pass

#class for node with value, activation sum, activation function, forward connection nodes,
# backward connection nodes, loss from function, the node loss,forward weights
class Node(metaclass=abc.ABCMeta):
    def __init__(self,act_func, value=0):
        self.value = value
        self.act_sum = 0
        self.act_func = act_func
        self.connections = [] # Node and weights e.g (node_h1, weight)
        self.back_connections = []
        self.func_loss = 0
        self.node_loss = 0
        self.weights = []

    def forward(self):
        for connection in self.connections:
            connection[0].act_sum += self.value*connection[1].w
    def addConnection(self,node,weight):
        self.connections.append((node,weight))
        self.weights.append(weight)
    def addBackConnection(self,node):
        self.back_connections.append(node)

    def addConnections(self, nodes,weights):
        for node,weight in zip(nodes,weights):
            self.addConnection(node,weight)
    def addBackConnections(self,nodes):
        for node in nodes:
            self.addBackConnection(node)
#calculate output based on activation function
    def activate_function(self):
        if self.act_func.lower() == "sigmoid":
            self.value = 1/(1+exp(-self.act_sum))
        elif self.act_func.lower() == "relu":
            self.value =self.act_sum if self.act_sum >0 else 0
        else:
            pass

    def calculate_function_loss(self):
        if self.act_func.lower() == "relu":
            self.func_loss = 1
        elif self.act_func.lower() == "sigmoid":
            self.func_loss = self.value*(1-self.value)
        else:
            pass
    def connect_forward(self,layer):
        ws = []
        for i in range(len(layer.nodes)):
            random_weight = Weight(random.uniform(0.001,0.5))
            ws.append(random_weight)
        self.addConnections(layer.nodes, ws)
    def connect_backward(self,layer):
        self.addBackConnections(layer.nodes)

    def update(self,lr):
        for n, weight in self.connections:
            weight.w -= lr*weight.loss
            weight.loss= 0

    def zero_loss(self):
        self.node_loss=0
        self.value = 0
        self.act_sum=0

    @abc.abstractmethod
    def show_node(self):
        pass

    @abc.abstractmethod
    def calculate_error(self):
        pass

    @abc.abstractmethod
    def calculate_node_loss(self):
        pass

class SimpleNode(Node):
    def __init__(self,act_func="", value=0):
        super().__init__(act_func,value)
    def calculate_error(self):
        pass
    def calculate_node_loss(self):
        pass
    def show_node(self):
        s = "The node with value {}\n".format(self.value)
        for idx, weight in enumerate(self.weights) :
            s += "\tthe weight {}: {}\n".format(idx + 1, weight.w)
        return s


class OutputNode(Node):
    def __init__(self,target,act_func="", value=0):
        super().__init__(act_func, value)
        self.target = target
        self.error= 0
# function to calculate squared loss of one output node
    def calculate_error(self):
        self.error =0.5* (self.target - self.value)**2
# function to calculate squared loss derivative
    def calculate_node_loss(self):
        self.node_loss = (self.target - self.value)*-1
    def show_node(self):
        s = "The output node with value {}\n".format(self.value)
        return s

# Layer class which takes values if it is input layer, number of nodes and activation function
# if it is hidden layer or output layer
class Layer(metaclass=abc.ABCMeta):
    def __init__(self,num_nodes,nodes=[],act_func=""):
        if len(nodes) == 0:
            ns = []
            for i in range(num_nodes):
                n = SimpleNode(act_func = act_func)
                ns.append(n)
            self.nodes = ns
        else:
            ns = []
            for i in range(num_nodes) :
                n = SimpleNode(value = nodes[i])
                ns.append(n)
            self.nodes = ns
        self.act_func = act_func
        self.bias = None

    def forward(self):
        for node in self.nodes:
            node.activate_function()
        for node in self.nodes:
            node.forward()
        if self.bias:
            self.bias.forward()
# calcualte losses for each node in the layer
    def calculate_losses(self):
        if (self.act_func != ""):
            for node in self.nodes:
                for n,weight in node.connections:
                    n.calculate_function_loss()
                    node.node_loss += n.node_loss*n.func_loss*weight.w
                    weight.loss = n.node_loss*n.func_loss*node.value
            if self.bias:
                for n,weight in self.bias.connections:
                    n.calculate_function_loss()
                    weight.loss = n.node_loss * n.func_loss * node.value
        else:
            for node in self.nodes:
                for n,weight in node.connections:
                    n.calculate_function_loss()
                    weight.loss = n.node_loss*n.func_loss*node.value
            if self.bias :
                for n, weight in self.bias.connections :
                    n.calculate_function_loss()
                    weight.loss = n.node_loss * n.func_loss * node.value

#function to connect each node to next layer
    def connect_forward(self,layer):
        for node in self.nodes:
            node.connect_forward(layer)
        if self.bias != None:
            self.bias.connect_forward(layer)

    def connect_backward(self,layer):
        for node in self.nodes:
            node.connect_backward(layer)

    def add_bias(self,val):
        b = SimpleNode(act_func="",value=val)
        self.bias = b

    def show_layer(self):
        s = "-" * 50+"\n"
        for node in self.nodes :
            s += node.show_node()
        if self.bias:
            s+= "The bias:\n"
            s+=self.bias.show_node()
        return s

    def update(self,lr):
        for node in self.nodes:
            node.update(lr)
        if self.bias!= None:
            self.bias.update(lr)

    def make_losses_zero(self):
        for node in self.nodes:
            node.zero_loss()



class SimpleLayer(Layer):
    def __init__(self,num_nodes,nodes=[],act_func=""):
        super().__init__(num_nodes,nodes,act_func)


class OutputLayer(Layer):
    def __init__(self,num_nodes,targets,nodes=[],act_func=""):
        if len(nodes) == 0 :
            ns = []
            for i in range(num_nodes) :
                n = OutputNode(targets[i],act_func = act_func)
                ns.append(n)
            self.nodes = ns
        else :
            ns = []
            for i in range(num_nodes) :
                n = OutputNode(targets[i],value = nodes[i])
                ns.append(n)
            self.nodes = ns
        self.total_error = 0
        self.act_func = act_func
        self.bias = None

    def output_nodes_losses(self):
        for node in self.nodes:
            node.calculate_node_loss()



class NeuralNetwork:
    def __init__(self,layers):
        self.layers = layers
        self.outputLayer = layers[-1]
        self.total_error = 0
# calculate total error
    def error(self):
        for node in self.outputLayer.nodes:
            node.calculate_error()
            self.total_error+=node.error

    def forward(self):
        for layer in self.layers:
            layer.forward()
#calculating losses in each layer for each weight and node
    def calculate_losses(self):
        self.error()
        self.outputLayer.output_nodes_losses()
        for i in range(len(self.layers)-2,-1,-1):
            self.layers[i].calculate_losses()

#function to connect layers for forward and backward propogation
    def connect_layers(self):
        for i in range(0,len(self.layers)-1):
            self.layers[i].connect_forward(self.layers[i+1])
        for i in range(len(self.layers)-1,0,-1):
            self.layers[i].connect_backward(self.layers[i-1])
    def show_network(self):
        s = "Neural Network: \n"
        for idx, layer in enumerate(self.layers):
            s+= "Layer {}\n".format(idx+1)
            s+=layer.show_layer()
        print(s)

    def update(self):
        lr = 0.001
        self.total_error = 0
        for i in range(0, len(self.layers) - 1) :
            self.layers[i].update(lr)
        for i in range(len(self.layers)-1,0,-1):
            self.layers[i].make_losses_zero()



if __name__ == "__main__":
    print("Hello, World")
    num_of_tests = 10
    for i in range(num_of_tests):
        inputLayer = SimpleLayer(2, nodes=[0.2, 0.6])
        inputLayer.add_bias(0.25)
        hiddenLayer1 = SimpleLayer(2, act_func="sigmoid")
        hiddenLayer1.add_bias(0.45)
        hiddenLayer2 = SimpleLayer(2, act_func="sigmoid")
        outputLayer = OutputLayer(2, targets=[0.1, 0.9], act_func="relu")

        layers = [inputLayer, hiddenLayer1, hiddenLayer2, outputLayer]
        nn = NeuralNetwork(layers)
        nn.connect_layers()

        losses = []

        for _ in range(10000):
            nn.forward()
            nn.calculate_losses()
            losses.append(nn.total_error)
            nn.update()

        x_axis = [i for i in range(1,10001,1)]
        plt.plot(x_axis,losses)
        plt.ylabel('loss')
        plt.xlabel('iteration')
        plt.savefig('./plot{}.png'.format(i+1))
        plt.clf()
        losses = []
    # print(losses)






