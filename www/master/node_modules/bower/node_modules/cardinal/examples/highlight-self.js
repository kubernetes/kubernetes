// This file will highlight itself using the default theme when run via: "node highlight-self"

var cardinal = require('..');

function highlight () {
  
  // Using the asynchronous highlightFile()
  // For synchronous highlighting use: highlightFileSync() - see highlight-self-hide-semicolons.js
  
  cardinal.highlightFile(__filename, { linenos: true }, function (err, res) {
    if (err) return console.error(err);
    console.log(res);
  });
}

highlight();
