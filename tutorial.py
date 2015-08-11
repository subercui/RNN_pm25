# -*- coding: utf-8 -*-
#So，now，Where do you train and when do gradient decent
import theano, theano.tensor as T
import numpy as np
import theano_lstm
import random

class Sampler:
    def __init__(self, prob_table):
        total_prob = 0.0
        if type(prob_table) is dict:
            for key, value in prob_table.items():#value读出概率
                total_prob += value
        elif type(prob_table) is list:
            prob_table_gen = {}
            for key in prob_table:
                prob_table_gen[key] = 1.0 / (float(len(prob_table)))#list没有初始概率，这里初始概率
            total_prob = 1.0
            prob_table = prob_table_gen
        else:
            raise ArgumentError("__init__ takes either a dict or a list as its first argument")
        if total_prob <= 0.0:
            raise ValueError("Probability is not strictly positive.")
        self._keys = []
        self._probs = []
        for key in prob_table:
            self._keys.append(key)
            self._probs.append(prob_table[key] / total_prob)
            
    def __call__(self):
        sample = random.random()
        seen_prob = 0.0
        for key, prob in zip(self._keys, self._probs):
            if (seen_prob + prob) >= sample:
                return key
            else:
                seen_prob += prob
        return key
samplers = {
    "punctuation": Sampler({".": 0.49, ",": 0.5, ";": 0.03, "?": 0.05, "!": 0.05}),
    "stop": Sampler({"the": 10, "from": 5, "a": 9, "they": 3, "he": 3, "it" : 2.5, "she": 2.7, "in": 4.5}),
    "noun": Sampler(["cat", "broom", "boat", "dog", "car", "wrangler", "mexico", "lantern", "book", "paper", "joke","calendar", "ship", "event"]),
    "verb": Sampler(["ran", "stole", "carried", "could", "would", "do", "can", "carry", "catapult", "jump", "duck"]),
    "adverb": Sampler(["rapidly", "calmly", "cooly", "in jest", "fantastically", "angrily", "dazily"])
    }
    
def generate_nonsense(word = ""):
    '''
    每次生成一语段，之后加标点，满足加的标点是句号，这就是一句话了，于是输出
    每一个语段有大概率包含完整的成分，但并不一定成分完整
    '''
    if word.endswith("."):
        return word
    else:
        if len(word) > 0:
            word += " "
        word += samplers["stop"]()
        word += " " + samplers["noun"]()
        if random.random() > 0.7:
            word += " " + samplers["adverb"]()
            if random.random() > 0.7:
                word += " " + samplers["adverb"]()
        word += " " + samplers["verb"]()
        if random.random() > 0.8:
            word += " " + samplers["noun"]()
            if random.random() > 0.9:
                word += "-" + samplers["noun"]()
        if len(word) > 500:
            word += "."
        else:
            word += " " + samplers["punctuation"]()
        return generate_nonsense(word)

def generate_dataset(total_size, ):
    sentences = []
    for i in range(total_size):
        sentences.append(generate_nonsense())
    return sentences

# generate dataset                
lines = generate_dataset(100)

### Utilities:
class Vocab:
    __slots__ = ["word2index", "index2word", "unknown"]
    
    def __init__(self, index2word = None):
        self.word2index = {}
        self.index2word = []
        
        # add unknown word:
        self.add_words(["**UNKNOWN**"])
        self.unknown = 0
        
        if index2word is not None:
            self.add_words(index2word)
                
    def add_words(self, words):
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.word2index)#等于word：value就在dict中的序号
                self.index2word.append(word)
                       
    def __call__(self, line):
        """
        Convert from numerical representation to words
        and vice-versa.
        """
        if type(line) is np.ndarray:
            return " ".join([self.index2word[word] for word in line])
        if type(line) is list:
            if len(line) > 0:
                if line[0] is int:
                    return " ".join([self.index2word[word] for word in line])
            indices = np.zeros(len(line), dtype=np.int32)
        else:
            line = line.split(" ")
            indices = np.zeros(len(line), dtype=np.int32)
        
        for i, word in enumerate(line):
            indices[i] = self.word2index.get(word, self.unknown)
            
        return indices
    
    @property
    def size(self):
        return len(self.index2word)
    
    def __len__(self):
        return len(self.index2word)
        
vocab = Vocab()
for line in lines:
    vocab.add_words(line.split(" "))#构成单词映射
    
def pad_into_matrix(rows, padding = 0):
    if len(rows) == 0:
        return np.array([0, 0], dtype=np.int32)
    lengths = map(len, rows)
    width = max(lengths)
    height = len(rows)
    mat = np.empty([height, width], dtype=rows[0].dtype)
    mat.fill(padding)
    for i, row in enumerate(rows):
        mat[i, 0:len(row)] = row
    return mat, list(lengths)

# transform into big numerical matrix of sentences:
numerical_lines = []
for line in lines:
    numerical_lines.append(vocab(line))#vocab(line)反悔了indices:数字编码好的这一个line
#print numerical_lines
numerical_lines, numerical_lengths = pad_into_matrix(numerical_lines)
#print numerical_lines
#print numerical_lengths

from theano_lstm import Embedding, LSTM, RNN, StackedCells, Layer, create_optimization_updates, masked_loss

def softmax(x):
    """
    Wrapper for softmax, helps with
    pickling, and removing one extra
    dimension that Theano adds during
    its exponential normalization.
    """
    return T.nnet.softmax(x.T)

def has_hidden(layer):
    """
    Whether a layer has a trainable
    initial hidden state.
    """
    return hasattr(layer, 'initial_hidden_state')

def matrixify(vector, n):
    return T.repeat(T.shape_padleft(vector), n, axis=0)

def initial_state(layer, dimensions = None):
    """
    Initalizes the recurrence relation with an initial hidden state
    if needed, else replaces with a "None" to tell Theano that
    the network **will** return something, but it does not need
    to send it to the next step of the recurrence
    """
    if dimensions is None:
        return layer.initial_hidden_state if has_hidden(layer) else None
    else:
        return matrixify(layer.initial_hidden_state, dimensions) if has_hidden(layer) else None
    
def initial_state_with_taps(layer, dimensions = None):
    """Optionally wrap tensor variable into a dict with taps=[-1]"""
    state = initial_state(layer, dimensions)
    if state is not None:
        return dict(initial=state, taps=[-1])
    else:
        return None

class Model:
    """
    Simple predictive model for forecasting words from
    sequence using LSTMs. Choose how many LSTMs to stack
    what size their memory should be, and how many
    words can be predicted.
    """
    def __init__(self, hidden_size, input_size, vocab_size, stack_size=1, celltype=LSTM):
        # declare model
        self.model = StackedCells(input_size, celltype=celltype, layers =[hidden_size] * stack_size)
        # add an embedding
        self.model.layers.insert(0, Embedding(vocab_size, input_size))
        # add a classifier:
        self.model.layers.append(Layer(hidden_size, vocab_size, activation = softmax))
        # inputs are matrices of indices,
        # each row is a sentence, each column a timestep
        self._stop_word   = theano.shared(np.int32(999999999), name="stop word")
        self.for_how_long = T.ivector()
        self.input_mat = T.imatrix()
        self.priming_word = T.iscalar()
        self.srng = T.shared_randomstreams.RandomStreams(np.random.randint(0, 1024))
        # create symbolic variables for prediction:(就是做一次整个序列完整的进行预测，得到结果是prediction)
        self.predictions = self.create_prediction()
        # create symbolic variable for greedy search:
        self.greedy_predictions = self.create_prediction(greedy=True)
        # create gradient training functions:
        self.create_cost_fun()
        self.create_training_function()
        self.create_predict_function()
        '''上面几步的意思就是先把公式写好'''
        
    def stop_on(self, idx):
        self._stop_word.set_value(idx)
        
    @property
    def params(self):
        return self.model.params
        
    def create_prediction(self,greedy=False):
        def step(idx,*states):
            new_hiddens=list(states)
            new_states=self.model.forward(idx,prev_hiddens = new_hiddens)
            if greedy:
                return
            else:
                return new_states#不论recursive与否，会全部输出
        
        inputs = self.input_mat[:,0:-1]
        num_examples = inputs.shape[0]
        if greedy:
            return
        else:
            outputs_info = [initial_state_with_taps(layer, num_examples) for layer in self.model.layers[1:]]
            result, _ = theano.scan(fn=step,
                                sequences=[inputs.T],
                                outputs_info=outputs_info)
                                

        return result[-1].transpose((2,0,1))
                                 
    def create_prediction(self, greedy=False):
        def step(idx, *states):
            # new hiddens are the states we need to pass to LSTMs
            # from past. Because the StackedCells also include
            # the embeddings, and those have no state, we pass
            # a "None" instead:
            new_hiddens = [None] + list(states)
            
            new_states = self.model.forward(idx, prev_hiddens = new_hiddens)#这一步更新!!!!，idx是layer_input
            #new_states是一个列表，包括了stackcells各个层的最新输出
            if greedy:
                new_idxes = new_states[-1]#即最后一层softmax的输出
                new_idx   = new_idxes.argmax()
                # provide a stopping condition for greedy search:
                return ([new_idx.astype(self.priming_word.dtype)] + new_states[1:-1]), theano.scan_module.until(T.eq(new_idx,self._stop_word))
            else:
                return new_states[1:]#除第0层之外，其他各层输出
       
        # in sequence forecasting scenario we take everything
        # up to the before last step, and predict subsequent
        # steps ergo, 0 ... n - 1, hence:
        inputs = self.input_mat[:, 0:-1]
        num_examples = inputs.shape[0]
        # pass this to Theano's recurrence relation function:
        
        # choose what gets outputted at each timestep:
        if greedy:
            outputs_info = [dict(initial=self.priming_word, taps=[-1])] + [initial_state_with_taps(layer) for layer in self.model.layers[1:-1]]
            result, _ = theano.scan(fn=step,
                                n_steps=200,
                                outputs_info=outputs_info)
        else:
            outputs_info = [initial_state_with_taps(layer, num_examples) for layer in self.model.layers[1:]]
            result, _ = theano.scan(fn=step,
                                sequences=[inputs.T],
                                outputs_info=outputs_info)
        '''就是这里sequences相当于每次把inputs的一个给到idx，改动这里使符合一次给多种的pm25形式'''
        '''outputs_info:就是说让scan把每回的输出重新传回fn的输入，而outputs_info就是第一回没有之前输出时，给入的值。于是output_info也暗示了这种回传的形式
        Second, if there is no accumulation of results, we can set outputs_info to None. This indicates to scan that it doesn’t need to pass the prior result to fn.'''
        
        '''The general order of function parameters to fn is:
            sequences (if any), prior result(s) (if needed), non-sequences (if any)
            not only taps should respect an order, but also variables, since this is how scan figures out what should be represented by what'''                                                  
        if greedy:
            return result[0]
        # softmaxes are the last layer of our network,指的就是result[-1]是softmax层
        # and are at the end of our results list:
#       print "res=", result
#        print "res eval=", result[-1].eval()
        
        return result[-1].transpose((2,0,1))
        # we reorder the predictions to be:
        # 1. what row / example
        # 2. what timestep
        # 3. softmax dimension
        
        
    '''def create_prediction(self, greedy=False):
        return result[-1].transpose((2,0,1))'''
                                 
    def create_cost_fun (self):
        # create a cost function that
        # takes each prediction at every timestep
        # and guesses next timestep's value:
        what_to_predict = self.input_mat[:, 1:]#每一句话除了第一个字符之后的所有字符，等于给了第一个，之后整句话是predict出来
        # because some sentences are shorter, we
        # place masks where the sentences end:
        # (for how long is zero indexed, e.g. an example going from `[2,3)`)
        # has this value set 0 (here we substract by 1):
        for_how_long = self.for_how_long - 1
        # all sentences start at T=0:
        starting_when = T.zeros_like(self.for_how_long)
                                 
        '''predict的是完整的句子后面的各个词,注意这个predictions只调用了一遍，那就是说这一遍就是一个mini batch了'''
        self.cost = masked_loss(self.predictions,
                                what_to_predict,
                                for_how_long,
                                starting_when).sum()
        
    def create_predict_function(self):
        self.pred_fun = theano.function(
            inputs=[self.input_mat],
            outputs =self.predictions,
            allow_input_downcast=True
        )
        
        self.greedy_fun = theano.function(
            inputs=[self.priming_word],
            outputs=T.concatenate([T.shape_padleft(self.priming_word), self.greedy_predictions]),
            allow_input_downcast=True
        )
                                 
    def create_training_function(self):
        updates, _, _, _, _ = create_optimization_updates(self.cost, self.params, method="adadelta")#这一步Gradient Decent!!!!
        self.update_fun = theano.function(
            inputs=[self.input_mat, self.for_how_long],
            outputs=self.cost,
            updates=updates,
            allow_input_downcast=True)
        
    def __call__(self, x):
        return self.pred_fun(x)
        
# construct model & theano functions:
model = Model(
    input_size=10,
    hidden_size=10,
    vocab_size=len(vocab),
    stack_size=1, # make this bigger, but makes compilation slow
    celltype=RNN # use RNN or LSTM
)
model.stop_on(vocab.word2index["."])

# train:
for i in range(10000):
    error = model.update_fun(numerical_lines, numerical_lengths)#这一个update_fun函数一调用，就把整个dataset：lines全跑了一遍，就相当于一次epoch learning   
    if i % 100 == 0:
        print("epoch %(epoch)d, error=%(error).2f" % ({"epoch": i, "error": error}))
    if i % 500 == 0:
        print(vocab(model.greedy_fun(vocab.word2index["the"])))
        
'''
1.predict：现在是一个词一predict下一个，改成一个输入结构，predict一个结果，之后再序列地读下一个输入结构就好了，满足应用需求
    这里输入结构就对应了词
    
2.pm25模型构想：之前一小时gfs六要素，当前时刻gfs六要素，之后一小时gfs六要素，当前时刻pm25，
    预测之后一小时pm25；
    之后将预测的pm25作为输入，再迭代下一个小时的pm25；
'''