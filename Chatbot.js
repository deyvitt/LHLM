import LanguageModel from './model.js';
import TextGenerator from './generation.js';
import { processInputText } from './model.js';
import AutonomousTrigger from './AutonomousTrigger.js'; // Import the AutonomousTrigger class

let model = new TextGenerator();


init();

// Function to get the user's input
function getUserInput() {
  return document.getElementById('input').value;
}

// Function to update the chat with a new message
function updateChat(sender, message) {
  let chat = document.getElementById('chat');
  chat.innerHTML += `<p><strong>${sender}:</strong> ${message}</p>`;
  chat.scrollTop = chat.scrollHeight; // Scroll to the bottom
}

// Function to clear the input field
function clearInput() {
  document.getElementById('input').value = '';
}

// Function to handle user input
async function handleUserInput(event) {
  event.preventDefault();

  // Get the input text
  let input = getUserInput();

  // Process the input text
  let inputData = await processInputText(input);

  // Create an AutonomousTrigger instance with the model's response
  let trigger = new AutonomousTrigger(response);

  // Use the AutonomousTrigger to perform an internet search
  let searchResults = await trigger.performSearch();

  // Understand the user's request and get the search terms
  let searchTerms = trigger.understandRequest();

  // Add the input text to the chat
  updateChat('You', input);

  // Generate the response
  let response = await model.processPrompt(inputData);

  // Add the response to the chat
  updateChat('Bot', response);

  // Clear the input
  clearInput();
}

document.getElementById('send-button').addEventListener('click', handleUserInput);
document.getElementById('chatbot-form').addEventListener('submit', handleUserInput);
 