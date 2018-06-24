import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST",one_hot=True)

data_num,input_num = mnist.train.images.shape
output_num = input_num
hidden_num = 500
latent_num = 20
batch_size = 200
epoch=5000

def weight_variable(input_num,hidden_num,latent_num,output_num):
	weights={"encoder_l1":tf.Variable(tf.truncated_normal([input_num,hidden_num],stddev=0.001)),
	"encoder_l2":tf.Variable(tf.truncated_normal([hidden_num,hidden_num],stddev=0.001)),
	"encoder_l3_mu":tf.Variable(tf.truncated_normal([hidden_num,latent_num],stddev=0.001)),
	"encoder_l3_log_var":tf.Variable(tf.truncated_normal([hidden_num,latent_num],stddev=0.001)),
	"decoder_l1":tf.Variable(tf.truncated_normal([latent_num,hidden_num],stddev=0.001)),
	"decoder_l2":tf.Variable(tf.truncated_normal([hidden_num,hidden_num],stddev=0.001)),
	"decoder_l3":tf.Variable(tf.truncated_normal([hidden_num,output_num],stddev=0.001))}

	return weights

def biases_variable(input_num,hidden_num,latent_num,output_num):
	biases={"encoder_l1":tf.Variable(tf.constant(0.,shape=[hidden_num])),
	"encoder_l2":tf.Variable(tf.constant(0.,shape=[hidden_num])),
	"encoder_l3_mu":tf.Variable(tf.constant(0.,shape=[latent_num])),
	"encoder_l3_log_var":tf.Variable(tf.constant(0.,shape=[latent_num])),
	"decoder_l1":tf.Variable(tf.constant(0.,shape=[hidden_num])),
	"decoder_l2":tf.Variable(tf.constant(0.,shape=[hidden_num])),
	"decoder_l3":tf.Variable(tf.constant(0.,shape=[output_num]))}

	return biases

def encoder(x,input_num,hidden_num,latent_num,weights,biases):
	l1=tf.nn.softplus(tf.add(tf.matmul(x,weights["encoder_l1"]),biases["encoder_l1"]))
	l2=tf.nn.softplus(tf.add(tf.matmul(l1,weights["encoder_l2"]),biases["encoder_l2"]))
	l3_mu=tf.add(tf.matmul(l2,weights["encoder_l3_mu"]),biases["encoder_l3_mu"])
	l3_log_var=tf.add(tf.matmul(l2,weights["encoder_l3_log_var"]),biases["encoder_l3_log_var"])

	return l3_mu,l3_log_var

def decoder(z,hidden_num,latent_num,weights,biases):
	l1=tf.nn.softplus(tf.add(tf.matmul(z,weights["decoder_l1"]),biases["decoder_l1"]))
	l2=tf.nn.softplus(tf.add(tf.matmul(l1,weights["decoder_l2"]),biases["decoder_l2"]))
	l3=tf.nn.sigmoid(tf.add(tf.matmul(l2,weights["decoder_l3"]),biases["decoder_l3"]))

	return l3


x_placeholder=tf.placeholder(tf.float32,shape=[None,input_num])
weights=weight_variable(input_num,hidden_num,latent_num,output_num)
biases=biases_variable(input_num,hidden_num,latent_num,output_num)

mu,log_var=encoder(x_placeholder,input_num,hidden_num,latent_num,weights,biases)
epsilon=tf.random_normal(tf.shape(log_var),0,1,dtype=tf.float32,name="epsilon_gaussian")
z=tf.add(mu,tf.multiply(tf.exp(0.5*log_var),epsilon))

x_reconst=decoder(z,hidden_num,latent_num,weights,biases)
#size=tf.shape(x_reconst)

#latent_loss=0.5*tf.reduce_sum(tf.square(mu)+tf.square(log_var)-tf.log(tf.square(log_var))-1,1)
latent_loss= 0.5 * tf.reduce_sum(-1 - log_var + tf.pow(mu, 2) + tf.exp(log_var), reduction_indices=1)
recong_loss=-tf.reduce_sum(x_placeholder * tf.log(1e-10 + x_reconst) + (1-x_placeholder) * tf.log(1e-10 + 1 - x_reconst), 1)
#recong_loss=tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_reconst, labels=x_placeholder), reduction_indices=1)

loss=tf.reduce_mean(latent_loss+recong_loss)
optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for step in range(1,epoch):
		batch=mnist.train.next_batch(batch_size)
		feed_dict={x_placeholder:batch[0]}
		_,error=sess.run([optimizer,loss],feed_dict=feed_dict)

		if step%1000==0:
			print("step{},loss{}".format(step,error))

	#print sess.run([size],feed_dict={x_placeholder:batch[0]})
        features = sess.run(mu, feed_dict=feed_dict)
	n=5
	canvas_orig=np.empty((28*n,28*n))
	canvas_recon = np.empty((28 * n, 28 * n))
        embed()
	for i in range(n):
		batch=mnist.test.next_batch(n)
		feed_dict={x_placeholder:batch[0]}
		g=sess.run(x_reconst,feed_dict=feed_dict)
		for j in range(n):
			canvas_orig[i*28:(i+1)*28,j*28:(j+1)*28]=batch[0][j].reshape([28,28])
			canvas_recon[i*28:(i+1)*28,j*28:(j+1)*28]=g[j].reshape([28,28])

print("Original Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_orig, origin="upper", cmap="gray")
#plt.show()
plt.savefig("./Original Images")

print("Reconstructed Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_recon, origin="upper", cmap="gray")
#plt.show()
plt.savefig("./Reconstructed Images")






