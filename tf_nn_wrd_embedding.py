from io import StringIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager , PDFPageInterpreter
from pdfminer.pdfpage import PDFPage

#
from  nltk.tokenize  import word_tokenize, sent_tokenize
import string
import numpy as np

import tensorflow as tf    
import numpy as np
import sys
#import pdf_text_extraction  as pdf
from sklearn.model_selection import train_test_split
import datetime



#helper function for pulling pdfdata 
def data_loader (path,max_pages=10):
    return_string = StringIO()
    device = TextConverter(PDFResourceManager(),return_string,codec='utf-8',laparams=LAParams())
    Interpreter = PDFPageInterpreter(PDFResourceManager(), device=device)
    filepath= open(path, 'rb')

    for page in PDFPage.get_pages(filepath, set(),
                                  maxpages=max_pages, caching= True,check_extractable=True):
        Interpreter.process_page(page)
        
    text_data = return_string.getvalue()
    filepath.close(),device.close(),return_string.close()

    return text_data
        

# word fucntion which makes words into one hot encoding
def tf_data(path):
    #data_loader function call from pdf
    text = data_loader(path)

    
    #verifyig that the characters are ascii
    text  =  ''.join( word for word in text if ord(word) < 128)

    #from  nltk.tokenize  import word_tokenize .Word tokenize fucntion tokenizes the text data we have got 
    vocab = word_tokenize(text)

    word_dict={}
    #indexing each word into a dictory
    for index, word in enumerate(word_tokenize(text)):
        word_dict[word]=index

    #from  nltk.tokenize  import sent_tokenize .sent_tokenize  used to tokenixe our text data . we ca give any delimiter inthis case we are
    #into default which is "."
    sentences = sent_tokenize(text)
    #tokenize each sentence for the words
    token_sents = list([word_tokenize(sent) for sent in sentences])


    n_gram_data=[]
    #using string.punctuation function to form a list of punctuations
    punctuation=list(string.punctuation)



    #now we have both token_sents , we are going to form a pair of two word embedding. Its upto the user to form any number of word embeddings
    #we are gng to word two word embeddings where we give the first word and try to predict the next work in the sequence for later.
    for sent in token_sents :
        for index,word in enumerate(sent):
            if word not in punctuation : #making sure that the words are not in the punctuations
                for _word in sent[max(index-1,0) : min(index+1,len(sent))+1] :   #looping each words in our sentences list for every value inthis list. Every value is again a list of tokenizes words
                    if _word != word: #make sure we are not mapping the same two words for our embeddings
                        n_gram_data.append([word,_word])
    print('word pairs done and vocab done')
    print('len_ngram'+str(len(n_gram_data)))
    print('len(vocab)'+ str(len(vocab)))
    x,y = np.zeros([len(n_gram_data),len(vocab)]), np.zeros([len(n_gram_data),len(vocab)]) #intializing np.zeros withe the shape of( len of word embeddings . len of our total words in text


    #one hot encoding for feeding to tf
    def one_hot_encoder(index, vocab_size):
        vector = np.zeros(vocab_size)
        vector[index] = 1
        return vector



    for i in range(len(n_gram_data)):
        x[i,:] = one_hot_encoder(word_dict[n_gram_data[i][0]],len(vocab)) # forming one hot of first word in two pair
        y[i,:] = one_hot_encoder(word_dict[n_gram_data[i][1]],len(vocab)) #label will be one hot of the subsequent word in two pair                           

    return x, y , len(vocab), word_dict,n_gram_data


#helper function for randomly taking data from our inputs
def next_batch(x,y,steps):
    ts_nan = np.random.randint(0,len(x)-steps)
    x = np.array(x[ts_nan:ts_nan+steps])
    y = np.array(y[ts_nan:ts_nan+steps])
    return x,y


#lets create the tensor flow model function
def tf_model (X_train , y_train,vocab_size):
    print(datetime.datetime.today())
    #keeping the embedding demension to 8 here from my inputs. from the tensoflow documentation "embedding_dimensions =  number_of_categories**0.25"
    #here our categories is 3976 as i have 3976 words so approximating the value as 8
    embedding_dim= 8

    #initialize the placeholder variables
    X = tf.placeholder(tf.float32, shape=(None, vocab_size))
    Y = tf.placeholder(tf.float32, shape=(None, vocab_size))

    #initialize the weights
    input_weights =tf.Variable(tf.random_normal([vocab_size, embedding_dim]))
    output_weights = tf.Variable(tf.random_normal([embedding_dim, vocab_size]))

    #intilize the bias variables
    input_biases =tf.Variable(tf.random_normal([embedding_dim]))
    output_biases = tf.Variable(tf.random_normal([vocab_size]))
    weights = {'hidden': tf.Variable(tf.random_normal([vocab_size, embedding_dim])),'output': tf.Variable(tf.random_normal([embedding_dim, vocab_size]))}
    biases = {'hidden': tf.Variable(tf.random_normal([embedding_dim])),'output': tf.Variable(tf.random_normal([vocab_size]))}
    #lets create our input and output layers
    input_layer = tf.add(tf.matmul(X, input_weights),input_biases)
    output_layer = tf.add(tf.matmul(input_layer,output_weights), output_biases)    
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer)) #lets use a softmax function as activation function
    adam_optimizer = tf.train.AdamOptimizer(0.01).minimize(error) #using adam optimizer for our gradients

    #initilaize our global variables
    init = tf.global_variables_initializer()
    with  tf.Session() as  sess:
        sess.run(init)
        
        for i in range(2000): #lets run for 2000 epochs to get decent result for our output layers
            x_batch,y_batch = next_batch(X_train,y_train,1000)
            sess.run(adam_optimizer, feed_dict = {X : x_batch, Y: y_batch})
            if i %100==0:
                print(str(i)+'done')

        #lets predict for some of the values as i need to form the input in the same format im considering that when in actual scenarios we used the same word of forming the word embeddings for our input to eb predicted and feed to the model
        #so lets take a chuck of our input # ideally use a different sent of data for predictions 
        y_pred=sess.run(output_layer,feed_dict={X:X_train[100:1100]}) #here our outlayer is the one which has the fully modelled tensor
    print(datetime.datetime.today())    
    
    #return y_pred, y_test
    return X_train[100:1100], y_pred  #lets return return both our inputs and predicted outputs


#lets run our model and predictions
path = 'test.pdf'# here giving an example use the path of pdf file which you want touse for 
x, y , len(vocab), word_dict,n_gram_data = tf_data(path)

#x is the input vector , y is the label vector , #word_dict #ngram_data for us to verify

steps =1000 # steps for taking the input data into model lets keep as 1000 for each iteration, we can lower it as well but need to give the same size for the predictions as well . please make sure
inputs, predictions =tf_model(x,y,steps)


#now the predictions
#the words which we predicted is in the shape (1000,3976) as we have 3976 words and the actual word we predicted in the word which has the maximum prbability value
# for exmaple
#inputs[1]  gives list of len 3976lets take the max which has the value 1 as we have one hot encoded them
# lets create simple function for that

def get_word (word_dict,input_vector):
    #for getting the exact colum element
    for i in range(len(input_vector)):
        if i == max(input_vector):
            ix= i
    #ix holds the  index of the word

    for i in range(word_dict):
        if word_dict[i] == ix :
            word = i
    return word

# lets pass our input and preds to see the words
input_word = get_word (word_dict, inputs[1])
predicted_word = get_word(word_dict, predictions[1])
actual_Word = get_word (word_dict, y[1])




    



