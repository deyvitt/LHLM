import * as tf from '@tensorflow/tfjs';
import { scaled_dot_product_attention, MultiHeadAttention } from './model.js';

class Attention extends tf.layers.Layer {
  constructor(units) {
    super({name: 'Attention'});
    this.W1 = tf.layers.dense({units: units});
    this.W2 = tf.layers.dense({units: units});
    this.V = tf.layers.dense({units: 1});
  }

  call(inputs) {
    // inputs is an array containing the encoder output and the decoder hidden state
    let [encoderOutput, decoderHidden] = inputs;

    // calculate the score
    let score = this.V.apply(tf.tanh(
      tf.add(this.W1.apply(encoderOutput), this.W2.apply(decoderHidden))
    ));

    // calculate the attention weights
    let attentionWeights = tf.softmax(score, 1);

    // calculate the context vector
    let contextVector = tf.sum(tf.mul(attentionWeights, encoderOutput), 1);

    return [contextVector, attentionWeights];
  }

  computeOutputShape(inputShape) {
    let [encoderOutputShape, decoderHiddenShape] = inputShape;
    return [[encoderOutputShape[0], encoderOutputShape[2]], encoderOutputShape];
  }
}

export default Attention; 