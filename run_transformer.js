import { Transformer } from './model.js';

export let transformer = new Transformer(512, 8, 2048);
console.log(transformer.countParams());
