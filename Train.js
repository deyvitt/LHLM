// Description: Train the model
import * as tf from '@tensorflow/tfjs';
import { TransformerBlock, EncoderLayer, Transformer } from './model.js';
import DataLoader from './DataLoader.js';
import fetch from 'node-fetch';
import prompt from 'prompt-sync';

const promptSync = prompt();

async function fetchData(title) {
  const response = await fetch(`https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&redirects=1&titles=${title}`);
  const data = await response.json();
  return data;
}

async function trainAndSaveModel(inputs, targets, modelSavePath) {
  // Define the model
  let model = new LanguageModel(inputDim, units, outputDim);

  // Train the model
  await model.fit(inputs, targets, epochs, batchSize);

// Define the model save path
const modelSavePath = '/Volumes/NO NAME/LHLM/my-chatbot';

// Save the model
await model.save(modelSavePath);
}

// Get the title of the Wikipedia page from the user
const title = promptSync('Enter the title of the Wikipedia page: ');

// Fetch data from the Wikipedia page
fetchData(title)
  .then(async (data) => { // Mark this function as async
    // Extract the text from the Wikipedia page
    let text = Object.values(data.query.pages)[0].extract;

    // Create a new DataLoader instance
    let dataLoader = new DataLoader();

    // Preprocess the text
    let {inputs, targets} = dataLoader.preprocess(text);

    // Train and save the model
    await trainAndSaveModel(inputs, targets, '/Volumes/NO NAME/LHLM/my-chatbot');
  })
  .catch(error => console.error(error));

document.getElementById('train-button').addEventListener('click', async () => {
  // Disable the button to prevent multiple clicks
  document.getElementById('train-button').disabled = true;

  // Update the status
  document.getElementById('status').textContent = 'Training...';

  // Load the data
  let dataLoader = new DataLoader();
  
  // Load data from file
  let file = document.getElementById('data-file').files[0];
  let {inputs: inputsFromFile, targets: targetsFromFile} = await dataLoader.load(file);

  // Load data from URL
  let {inputs: inputsFromUrl, targets: targetsFromUrl} = await dataLoader.loadFromUrl('https://example.com/data.json');

  // Concatenate inputs and targets from file and URL
  let inputs = tf.concat([inputsFromFile, inputsFromUrl]);
  let targets = tf.concat([targetsFromFile, targetsFromUrl]);

  // Train and save the model
  await trainAndSaveModel(inputs, targets, 'localstorage://my-model');

  // Update the status
  document.getElementById('status').textContent = 'Training complete';

  // Enable the button
  document.getElementById('train-button').disabled = false;
});