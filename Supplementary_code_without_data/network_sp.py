"""network_sp.py
~~~~~~~~~~~~~~

Sparse network without dendritic normalisation

"""

#### Libraries
import numpy as np

class LogLikelihoodCost(object):

    @staticmethod
    def fn(a, y):
        return -np.log(a[np.nonzero(y)])

    @staticmethod
    def delta(z, a, y):       
        return (a-y)


#### Main Network class
class Network(object):

    def __init__(self, sizes, con_prob, con_delt, cost=LogLikelihoodCost):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [784, 30, 10]
        then it would be a three-layer network, with the first layer
        containing 784 neurons, the second layer 30 neurons, and the
        third layer 10 neurons. 'con_prob' gives the connection probability and 'con_delt' the proportion of weakest contacts which are excised at each epoch under the SET algorithm. """ 
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weight_initializer(con_prob)
        self.cost=cost
        self.con_delt=con_delt

    def weight_initializer(self,con_prob):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        
        self.weights=[]
        for l_ind in range(len(self.sizes)-2):
            tot_sz=self.sizes[l_ind]*self.sizes[l_ind+1]
            iwght=np.zeros([tot_sz,1])
            n_allowed=int(np.round(con_prob*tot_sz))
            ind_allowed=np.random.choice(tot_sz,n_allowed,replace=False)
            vals_allowed=np.random.randn(n_allowed,1)
            iwght[ind_allowed]=vals_allowed
            
            self.weights.append(np.reshape(iwght,[self.sizes[l_ind+1],self.sizes[l_ind]]))
        
        iwght=np.random.randn(self.sizes[-1],self.sizes[-2])
        self.weights.append(iwght)

    def feedforward(self, a):        
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = sigmoid(np.dot(w, a)+b)
        
        lS=self.num_layers-2
        w=self.weights[lS]
        b=self.biases[lS]
        a=softmax(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            evaluation_data=None,
            show_progress=True):
        """Train the neural network using stochastic gradient
        descent and the SET to regulate sparsity.  
        """

        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        
        con_delt=self.con_delt
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, len(training_data))            
             
            # SET step
            for l_ind in range(self.num_layers-2):
                wght=self.weights[l_ind]
                totsz=np.size(wght)
                n_con=np.count_nonzero(wght)
                n_change=round(n_con*con_delt)
                w_vals=wght[np.nonzero(wght)]
                w_abs=np.sort(np.abs(w_vals))
                w_thresh=w_abs[n_change]
                wght[np.abs(wght)<w_thresh]=0            
                            
                if j<epochs: # Add random new weights if not last run
                    n_con_2=np.count_nonzero(wght)
                    n_change=n_con-n_con_2
                    
                    pos_locs=np.nonzero(wght==0)
                    pos_rws=pos_locs[0]
                    pos_cls=pos_locs[1]
                
                    new_ws=np.random.randn(n_change)
                    new_inds=np.random.choice(np.size(pos_rws),n_change,replace=False)
                
                    wght[pos_rws[new_inds],pos_cls[new_inds]]=new_ws
                    
                self.weights[l_ind]=wght
             
            cost = self.total_cost(training_data)
            training_cost.append(cost)
            accuracy = self.accuracy(evaluation_data)
            evaluation_accuracy.append(accuracy)
            if show_progress:
                print("Epoch %s training complete" % j)                
                print("Cost on training data: {}".format(cost))               
                print("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data))                               
                
        return evaluation_accuracy, training_cost

    def update_mini_batch(self, mini_batch, eta, n):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]        
            
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        self.weights = [1-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        
       

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] 
        zs = [] 
        
        
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
             z = np.dot(w, activation)+b
             zs.append(z)
             activation = sigmoid(z)
             activations.append(activation)            
        
        lS=len(self.sizes)-2
        w=self.weights[lS]
        b=self.biases[lS]
        z = np.dot(w, activation)+b                
        zs.append(z)
        activation = softmax(z)
        activations.append(activation)
     
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            msk=np.zeros(np.shape(self.weights[-l]))
            msk[np.nonzero(self.weights[-l])]=1            
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())*msk
        return (nabla_b, nabla_w)

    def accuracy(self, data,):       
        results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data):     
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            cost += self.cost.fn(a, y)/len(data)
        return cost

#### Miscellaneous functions

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def softmax(z):
    rawexp=np.exp(z)
    S=rawexp.sum()
    return rawexp/S

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

