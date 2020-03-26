#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# our "R"
def get_distribution_sampler(mu, sigma):
    # use a random normal distribution
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1,n)))

# our "I"
def get_generator_input_sampler():
    # use a uniform distribution
        # this makes it harder for G to copy R
    return lambda m, n: torch.rand(m, n)

# our "G"
'''
feed forward nn with 2 hidden layers (3 maps)
I -> G -> ??? -> mimic R
    - it never has to look at R
'''
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Generator, self).__init__()
        #input -> map1[f(x)] -> map2[f(x)] -> map3 -> output
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f

    # feed forward
    def forward(self, x):
        x = self.map1(x)
        x = self.f(x)
        x = self.map2(x)
        x = self.f(x)
        x = self.map3(x)
        return x

# our "D"
'''
- feed forward nn with 2 hidden layers (3 maps)
- sigmoid activation function

Input (R || G) -> 0 - 1 (Truthiness)
'''
class Discriminatior(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Discriminatior, self).__init__() 
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, x):
        x = self.f(self.map1(x))
        x = self.f(self.map2(x))    
        return self.f(self.map3(x))


# training loop
for epoch in range(num_epochs):
    '''
    Training D:

    push both types of data through G then calcuate the gradient (.backward())
        D's gradient is updated, G's in untouched
    '''
    for d_index in range(d_steps):
        D.zero_grad()

        # Train D on the real data
        d_real_data = Variable(d_sampler(d_input_size))
        d_real_decision = D(preprocess(d_real_data))
        d_real_error = criterion(d_real_decision, Variable(torch.ones([1,1]))) # 1's = true
        d_real_error.backward() # compute and store gradient -- backfeeding (?)

        #train D on the fake data
        d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
        d_fake_data = G(d_gen_input).detach() # detach to avoid D training on this
        d_fake_decision = D(preprocess(d_fake_data.t()))
        d_fake_error = criterion(d_fake_decision, Variable(torch.zeros([1,1])))
        d_fake_error.backward()

        # step forward one
        d_optimizer.step()
        dre, dfe = extract(d_real_error)[0], extract(d_fake_error)[0]
    ''' Training G:
        do the same thing as above
        also run G's output through D (staticaly)
    '''
    for g_index in range(g_steps):
        G.zero_grad()

        gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
        g_fake_data = G(gen_input)
        dg_fake_decision = D(preprocess(g_fake_data.t()))
        g_error = criterion(dg_fake_decision, Variable(torch.ones([1,1]))) # make G think it's genuine

        g_error.backward()
        g_optimizer.stop()
        ge = extract(g_error)[0]

