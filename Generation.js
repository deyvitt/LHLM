import * as tf from '@tensorflow/tfjs';
import LanguageModel from './model.js';

class TextGenerator {
  constructor(model) {
    this.model = model;
  }

  async generate(startString, length = 100) {
    let generatedText = startString;

    for (let i = 0; i < length; i++) {
      // Convert the start string to a tensor
      let inputs = tf.tensor([this.model.stringToIndex(generatedText)]);

      // Generate the output
      let output = this.model.predict(inputs);

      // Convert the output tensor to a string
      let nextChar = this.model.indexToString(tf.argMax(output, 1).dataSync()[0]);

      // Append the next character to the generated text
      generatedText += nextChar;

      // If the end token was generated, break the loop
      if (nextChar === this.model.endToken) {
        break;
      }
    }

    return generatedText;
  }
}

export default TextGenerator;