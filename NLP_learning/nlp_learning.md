## xavier_uniform() and xavier_transform()
usually use the xavier_uniform_() function. That function has an optional gain parameter that is related to the activation function used on the layer. The idea is best explained using a code example. Xavier computes the two range endpoints automatically based on the number of input nodes (“fan-in”) and output nodes (“fan-out”) to the layer
