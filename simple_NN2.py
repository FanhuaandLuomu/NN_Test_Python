#coding:utf-8
# 创建3->4->1的三层MLP

from numpy import exp,array,random,dot

class NeuronLayer():
	def __init__(self,number_of_neurons,number_of_inputs_per_neuron):
		self.synaptic_weights=2*random.random((\
			number_of_inputs_per_neuron,number_of_neurons))-1

class NeuralNetWork():
	def __init__(self,layer1,layer2):
		self.layer1=layer1
		self.layer2=layer2

	# sigmoid 函数 
	def __sigmoid(self,x):
		return 1/(1+exp(-x))

	# sigmoid 的梯度
	def __sigmoid_derivative(self,x):
		return x*(1-x)

	# 计算
	def think(self,inputs):
		output_from_layer1=self.__sigmoid(dot(inputs,
			self.layer1.synaptic_weights))
		output_from_layer2=self.__sigmoid(dot(output_from_layer1,
			self.layer2.synaptic_weights))
		return output_from_layer1,output_from_layer2

	# 通过训练 调整权重
	def train(self,training_set_inputs,training_set_outputs,
			number_of_training_iterations):
		for iteration in xrange(number_of_training_iterations):
			# 将整个训练集传递给神经网络
			output_from_layer1,output_from_layer2=self.think(training_set_inputs)

			# 计算第二层的误差
			layer2_error=training_set_outputs - output_from_layer2
			layer2_delta=layer2_error*self.__sigmoid_derivative(output_from_layer2)

			# 计算第一层的误差 得到第一层对第二层的影响
			layer1_error=layer2_delta.dot(self.layer2.synaptic_weights.T)
			layer1_delta=layer1_error*self.__sigmoid_derivative(output_from_layer1)

			# 计算权重调整量
			layer1_adjustment=training_set_inputs.T.dot(layer1_delta)
			layer2_adjustment=output_from_layer1.T.dot(layer2_delta)

			# 调整权重
			self.layer1.synaptic_weights+=layer1_adjustment
			self.layer2.synaptic_weights+=layer2_adjustment

	def print_weights(self):
		print "Layer 1(4 neurons,each with 3 inputs):"
		print self.layer1.synaptic_weights
		print "Layer 2(1 neuron,with 4 inputs):"
		print self.layer2.synaptic_weights


if __name__ == '__main__':
	# 设定随机数种子
	random.seed(1)

	# 输入层  4 个神经元  3 个输入
	layer1=NeuronLayer(4,3)

	# 隐藏层  1 个神经元 4 个输入
	layer2=NeuronLayer(1,4)

	# 组合成网络
	neural_network=NeuralNetWork(layer1,layer2)

	print 'stage 1) 随机初始化权重'
	neural_network.print_weights()

	# 训练样本 7 个样本 
	training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
	training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

	# 用训练集训练网络
	neural_network.train(training_set_inputs,training_set_outputs,60000)

	print 'stage 2)训练后的权重：'
	neural_network.print_weights()

	# 测试
	print "stage 3)[1,1,0]->?"
	hidden_state,output=neural_network.think(array([1,1,0]))
	print hidden_state
	print output
	print 1 if output>0.5 else 0