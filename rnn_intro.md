## Introduction to Recurrent Neural Network
In the previous chapters, we presented *simple* [Feedforward Neural Networks](https://medium.com/towards-data-science/deep-learning-feedforward-neural-network-26a6705dbdc7) that varied in size and purpose. These type of networks work well on structured (fact based) data where both event order and location relative to others records are irrelevant. However, this, as you might imagine, is not always the case. Thus [other types of networks](http://colah.github.io/posts/2015-09-NN-Types-FP/) should be used.

For example, consider images, where every pixel has a specific location. This pixel location by itself does not provide us with a useful information. In order to understand the image, the pixel's neighbours and their neighbours have to be considered as well. For these type of problems, the [Convolutional Neural Networks](http://cs231n.github.io/convolutional-networks/) (CNN, *not a news agency!*) are used, as they are made to learn from the information that is contained in the pixel and also around it. In this tutorial, we will not discuss this type of networks as they are not often used on the structured data, like, patient records, transaction records, etc.

The other type of network that is gaining popularity is the [Recurrent Neural Network](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) (RNN).

### Recurrent Neural Network
Recurrent Neural Networks are called recurrent because they perform the same computations for all elements in a sequence of inputs. RNNs are becoming very popular due to their wide utility. Often used example is language, where a word order is as important as the word by itself. Similarly, if we consider a patient medical history, in order to make a *good* prediction for some condition, it is not only important to know what the patient had before but also when it occurred.

The reason that makes recurrent networks exciting is that they allow us to operate over sequences of vectors or event timelines. This means that we can have sequences in the input or the output, or in the most general case both. Having sequences as inputs or outputs yields that in order to learn from this data efficiently, the network has to remember what it has seen. For that reason, RNN has an internal state, that is like a memory. As the RNN devours a sequence, the essential information about the sequence is maintained in this memory unit (internal state), and updated at each time-step. To learn more about RNNs, you can read these [series of articles](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) on RNN.

However, if a sequence is long, in practice, the internal state has a very difficult time to store all the information.  The problem lies, especially, with storing the beginning of the sequence. Due to this when we perform a back-propagation in order to update weights, the computed gradients become progressively smaller or larger as we come closer to the beginning of the sequence. This phenomenon is called the [Exploding and Vanishing Gradient](http://web.stanford.edu/class/cs224n/lecture_notes/cs224n-2017-notes5.pdf) problem for RNNs. To solve the problem of exploding gradients, a simple heuristic solution that clips gradients to a small number whenever they explode. That is, whenever they reach a certain threshold, they are set back to a small number. Whereas to solve the problem of vanishing gradients, the following two techniques are often used:

+ instead of initializing weights randomly, we start off from an identity matrix initialization,
+ instead of using the sigmoid activation function we can use the Rectified Linear Units (ReLU) function. The derivative for the ReLU is either 0 or 1. This way, gradients would flow through the neurons whose derivative is 1 without getting attenuated while propagating back through time-steps.

Another approach is to use more sophisticated units, such as [**LSTM** (Long Short-Term Memory)](https://en.wikipedia.org/wiki/Long_short-term_memory) or [**GRU** (Gated Recurrent Unit)](https://en.wikipedia.org/wiki/Gated_recurrent_unit). These units were explicitly designed to prevent the problem of exploding and vanishing gradients as well as improve long term memory of the RNN.
> Note: Throughout this tutorial, we are going to use only LSTM units.








### References
+ [Andrej Karpathy blog](http://karpathy.github.io/)
+ [colah's blog](http://colah.github.io/)
+ [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)
+ [CS231n: Convolutional Neural Networks for Visual Recognition.](http://cs231n.github.io/)
+ [Deep Learning: Feedforward Neural Network](https://medium.com/towards-data-science/deep-learning-feedforward-neural-network-26a6705dbdc7)
+ [Recurrent Neural Networks Tutorial, Part 1 – Introduction to RNNs](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)
+ Wikipedia articles on [Long Short-Term Memory](https://en.wikipedia.org/wiki/Long_short-term_memory) and [Gated Recurrent Unit](https://en.wikipedia.org/wiki/Gated_recurrent_unit)
