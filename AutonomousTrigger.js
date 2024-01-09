import AttentionModel from './Attention.js';
import SecondAttentionModel from './SecondAttention.js';

export default class AutonomousTrigger {
  constructor(input) {
    this.input = input;
    this.firstModel = new AttentionModel();
    this.secondModel = new SecondAttentionModel();
  }

// Method to identify key words in the input
identifyKeyWords() {
    let input = this.input;
    let output;
  
    // Loop until the output from the second attention model is equal to the original input
    while (input !== this.input) {
      // Run the input through the first attention model
      let firstOutput = this.firstModel.processInput(input);
  
      // Run the first output through the second attention model
      output = this.secondModel.processInput(firstOutput);
  
      // Update the input for the next iteration
      input = output;
    }
  
// Your code here to interpret the output and identify key words
let words = output.split(' ');
let regex = /code|attention|mechanism/;
let keyWords = words.filter(word => regex.test(word));
    return output;
  }
  // Method to trigger a search based on the key words

  async triggerSearch() {
    const keyWords = this.identifyKeyWords();

    // Use the Bing Search API to search for the key words
    const response = await axios.get('https://api.bing.microsoft.com/v7.0/search', {
      params: {
        q: keyWords.join(' ')
      },
      headers: {
        'Ocp-Apim-Subscription-Key': 'your-bing-search-api-key'
      }
    });

    // Extract the search results from the response
    const searchResults = response.data.webPages.value.map(page => ({
      name: page.name,
      url: page.url,
      snippet: page.snippet
    }));


  return searchResults;  
  }
}