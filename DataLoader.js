import * as tf from '@tensorflow/tfjs';
import https from 'https';

class DataLoader {
  async load(file) {
    // Create a new FileReader instance
    let reader = new FileReader();

    // Wrap the FileReader readAsText method in a promise
    let promise = new Promise((resolve, reject) => {
      reader.onload = event => resolve(event.target.result);
      reader.onerror = error => reject(error);
    });

    // Read the file
    reader.readAsText(file);

    // Wait for the file to be read
    let data = await promise;

    // Preprocess the data
    // Split the string into an array of numbers
    let numbers = data.split('\n').map(Number);
    // Convert the array of numbers into a tensor
    let inputs = tf.tensor(numbers);
    let targets = tf.tensor(numbers);

    return {inputs, targets};
  }

  async loadFromUrl(url) {
    let data = await new Promise((resolve, reject) => {
      https.get(url, (resp) => {
        let data = '';

        // A chunk of data has been received.
        resp.on('data', (chunk) => {
          data += chunk;
        });

        // The whole response has been received.
        resp.on('end', () => {
          resolve(data);
        });

      }).on("error", (err) => {
        console.log("Error: " + err.message);
        reject(err);
      });
    });

    // Preprocess the data
    // Split the string into an array of numbers
    let numbers = data.split('\n').map(Number);
    // Convert the array of numbers into a tensor
    let inputs = tf.tensor(numbers);
    let targets = tf.tensor(numbers);

    return {inputs, targets};
  }
}

export default DataLoader;