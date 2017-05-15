#coding:utf-8
# 用python 搭建简易的神经网络
# 3*1

# example 1

from numpy import exp,array,random,dot

training_set_inputs=array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
training_set_outputs=array([[0,1,1,0]]).T

print training_set_inputs.shape
print training_set_outputs.shape

random.seed(1)

synaptic_weights=2*random.random((3,1))-1  # -1~1

for i in xrange(10000):
	output=1/(1+exp(-(dot(training_set_inputs,synaptic_weights))))
	synaptic_weights+=dot(training_set_inputs.T,
			(training_set_outputs-output)*output*(1-output))
	# print synaptic_weights
print synaptic_weights

print 1/(1+exp(-(dot(array([1,0,0]),synaptic_weights))))

################

class NeuralNetwork():
	def __init__(self):
		# 随机数种子
		random.seed(1)

		# 对单个神经元建模 3个输入连接 一个输出  -1~-  均值 0
		self.synaptic_weights=2*random.random((3,1))-1

	# Sigmoid函数
	def __sigmoid(self,x):
		return 1/(1+exp(-x))

	# Sigmoid 函数的导数 梯度
	def __sigmoid_derivative(self,x):
		return x*(1-x)

	def think(self,inputs):
		# 把输入传递给神经网络
		return self.__sigmoid(dot(inputs,self.synaptic_weights))

	def train(self,training_set_inputs,training_set_outputs,
		number_of_training_iterations):
		for iteration in xrange(number_of_training_iterations):
			# 将训练集导入神经网络
			output=self.think(training_set_inputs)
			# 计算误差
			error=training_set_outputs-output

			# 将误差 输入 sigmoid的梯度 相乘
			adjustment=dot(training_set_inputs.T,\
					error*self.__sigmoid_derivative(output))

			# 调整权重
			self.synaptic_weights+=adjustment

if __name__ == '__main__':
	# 初始化神经1网络
	neural_network=NeuralNetwork()
	print '随机初始化权值'
	print neural_network.synaptic_weights

	# 训练集 四个样本 每个有3个输入 1个输出
	training_set_inputs=array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
	training_set_outputs=array([[0,1,1,0]]).T

	# 用训练集训练网络
	neural_network.train(training_set_inputs,training_set_outputs,10000)

	print '训练后的权重'
	print neural_network.synaptic_weights

	# 测试
	print '[1,0,1]->?'
	print neural_network.think(array([1,0,0]))
