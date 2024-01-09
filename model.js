import * as tf from '@tensorflow/tfjs';

let loadedModel;

async function loadFile(filePath) {
// Load the model
loadedModel = await tf.loadLayersModel(filePath);
}

loadFile('file:///Volumes/NO NAME/LHLM/my-chatbot/model.json').then(() => {
  // After the model is loaded, we can use it for prediction

  // Instantiate the Transformer
  let transformer = new Transformer(d_model, num_heads, dff);

  // Import the tokenizer
  const tokenizer = require('@tensorflow-models/universal-sentence-encoder');

  // Instantiate the tokenizer
  const use = new tokenizer.UniversalSentenceEncoder();
  
  // Export a function that takes the input text as a parameter
  module.exports.processInputText = async function(inputText) {

    // Tokenize and encode the text
    const tokens = await use.encode([inputText]);

    // The tokens variable is now a tensor that represents your input text
    let inputData = tokens;

    // Process the input data with the Transformer
    let outputData = transformer.call(inputData, training=true, mask=null);

    // The outputData is now a tensor representing the Transformer's output
    // You can use outputData for further processing or print it out
    console.log(outputData);
  };

  }).catch(error => console.error(error));

async function makePrediction(inputData) {
  const prediction = loadedModel.predict(inputData);
  console.log(prediction);
}

function scaled_dot_product_attention(query, key, value, mask) {
  let matmul_qk = tf.matMul(query, key, false, true);

  // scale matmul_qk
  let dk = tf.tensor(key.shape[-1], 'float32');
  let scaled_attention_logits = matmul_qk.div(tf.sqrt(dk));

  // add the mask to the scaled tensor.
  if (mask !== undefined) {
    scaled_attention_logits = scaled_attention_logits.add(mask.mul(-1e9));
  }

  // softmax is normalized on the last axis (seq_len_k) so that the scores
  // add up to 1.
  let attention_weights = tf.softmax(scaled_attention_logits, -1);

  let output = tf.matMul(attention_weights, value);

  return output;
}

class MultiHeadAttention {
  // Implementation of multi-head attention goes here
    constructor(d_model, num_heads) {
      this.num_heads = num_heads;
      this.d_model = d_model;
  
      this.depth = Math.floor(this.d_model / this.num_heads);
  
      this.wq = tf.layers.dense({units: d_model});
      this.wk = tf.layers.dense({units: d_model});
      this.wv = tf.layers.dense({units: d_model});
  
      this.dense = tf.layers.dense({units: d_model});
    }

    split_heads(x, batch_size) {
      x = tf.reshape(x, [batch_size, -1, this.num_heads, this.depth]);
      return tf.transpose(x, [0, 2, 1, 3]);
    }
  
    call(value, key, query, mask) {
      let batch_size = tf.shape(query)[0];
  
      query = this.wq.apply(query);
      key = this.wk.apply(key);
      value = this.wv.apply(value);
  
      query = this.split_heads(query, batch_size);
      key = this.split_heads(key, batch_size);
      value = this.split_heads(value, batch_size);
  
      let scaled_attention = scaled_dot_product_attention(query, key, value, mask);
      scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3]);
  
      let concat_attention = tf.reshape(scaled_attention, [batch_size, -1, this.d_model]);
      let output = this.dense.apply(concat_attention);
      return output;
    }

    countParams() {
      let params = this.d_model * this.num_heads;
      console.log('MultiHeadAttention params:', params);
      return params;
    }
  }
  // Implement the FeedForwardNetwork class
  class FeedForwardNetwork {
    constructor(d_model, dff) {
      this.d_model = d_model;
      this.dff = dff;

      this.dense1 = tf.layers.dense({units: dff, activation: 'relu'});
      this.dense2 = tf.layers.dense({units: d_model});
    }
  
    call(x) {
      let out = this.dense1.apply(x);
      out = this.dense2.apply(out);
      return out;
    }

    countParams() {
      let params = this.d_model * this.dff;
      console.log('FeedForwardNetwork params:', params);
      return params;    }
  }

class TransformerBlock {
  constructor(d_model, num_heads, dff, rate) {
    this.attention = new MultiHeadAttention(d_model, num_heads, dff, rate);
    this.norm1 = tf.layers.layerNormalization();
    this.norm2 = tf.layers.layerNormalization();
    this.ffn = new FeedForwardNetwork(d_model, dff);

    this.dropout1 = tf.layers.dropout({rate: rate});
    this.dropout2 = tf.layers.dropout({rate: rate});
  }

  call(x, training, mask) {
    let attn_output = this.attention.call([x, x, x, mask]);  // Self attention (V == Q == K)
    let out1 = this.dropout1.apply(attn_output, training=training).add(x);

    let ffn_output = this.ffn.call(this.norm1.apply(out1));
    let out2 = this.dropout2.apply(ffn_output, training=training).add(out1);

    return this.norm2.apply(out2);
  }
}

class EncoderLayer {
  constructor(d_model, num_heads, dff) {
    this.multiHeadAttention = new MultiHeadAttention(d_model, num_heads, dff);
    this.feedForwardNetwork = new FeedForwardNetwork(d_model, num_heads, dff);
    this.layers = [this.multiHeadAttention, this.feedForwardNetwork];
  }

  call(x, training, mask) {
    let out1 = this.block1.call(x, training, mask);
    let out2 = this.block2.call(out1, training, mask);
    // Pass through additional blocks as needed
    return out2;
  }
  countParams() {
    let totalParams = 0;
    for (let layer of this.layers) {
      totalParams += layer.countParams();
    }
    return totalParams;
  }
}

class DecoderLayer {
  constructor(d_model, num_heads, dff, rate) {
    this.multiHeadAttention1 = new MultiHeadAttention(d_model, num_heads, dff, rate);
    this.multiHeadAttention2 = new MultiHeadAttention(d_model, num_heads, dff, rate);
    this.feedForwardNetwork = new FeedForwardNetwork(d_model, dff);
    
    this.norm1 = tf.layers.layerNormalization();
    this.norm2 = tf.layers.layerNormalization();
    this.dropout1 = tf.layers.dropout({rate: rate});
  }

  call(x, training, mask) {
    let out1 = this.block1.call(x, training, mask);
    let out2 = this.block2.call(out1, training, mask);
    // Pass through additional blocks as needed
    return out2;
  }
  countParams() {
  return this.multiHeadAttention1.countParams() + this.multiHeadAttention2.countParams() + this.feedForwardNetwork.countParams();
  }
}

export class Transformer {
  constructor(d_model, num_heads, dff) {
    this.encoder = new EncoderLayer(d_model, num_heads, dff);
    this.decoder = new DecoderLayer(d_model, num_heads, dff);
    this.layers = [this.encoder, this.decoder];
  }

    countParams() {
      let totalParams = 0;
      for (let layer of this.layers) {
        let layerParams = layer.countParams();
        totalParams += layerParams;
      }
      return totalParams;
    }

  call(x, training, mask) {
    let enc_output = this.encoder.call(x, training, mask);
    let dec_output = this.decoder.call(enc_output, training, mask);
    return dec_output;
  }

  async processPrompt(prompt) {
    // Preprocess the prompt
    let processedPrompt = this.preprocessPrompt(prompt);

    // Pass the processed prompt through the model
    let modelOutput = this.call(processedPrompt);

    // Postprocess the model's output
    let response = this.postprocessOutput(modelOutput);

    return response;
  }

  preprocessPrompt(prompt) {
  // Split the prompt into words
  let words = prompt.split(' ');

  // Replace each word with its corresponding token
  let tokens = words.map(word => this.wordToToken[word]);

  // Convert the tokens to a tensor
  let tokenTensor = tf.tensor(tokens);

  // Expand the dimensions of the tensor to match the input shape expected by the model
  let inputTensor = tokenTensor.expandDims(0);

  return inputTensor;
  }

  postprocessOutput(output) {
    // Remove the extra dimension added by the preprocessing step
    let tokenTensor = output.squeeze([0]);
  
    // Convert the tensor to an array of tokens
    let tokens = Array.from(tokenTensor.dataSync());
  
    // Replace each token with its corresponding word
    let words = tokens.map(token => this.tokenToWord[token]);
  
    // Join the words into a sentence
    let response = words.join(' ');
  
    return response;
  }
}
export { TransformerBlock, EncoderLayer };
