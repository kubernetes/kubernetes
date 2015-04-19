// This file will highlight the passed code using the custom theme when run via: "node highlight-string"

var cardinal = require('..');

var code = '' + 

function add (a, b) {
  var sum = a + b;
  return sum;
} + 
  
'';

console.log(cardinal.highlight(code));
