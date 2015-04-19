"use strict";

module.exports = function(label) {
  var debug;

  if (process.env.NODE_DEBUG && /\blog4js\b/.test(process.env.NODE_DEBUG)) {
    debug = function(message) { 
      console.error('LOG4JS: (%s) %s', label, message); 
    };
  } else {
    debug = function() { };
  }

  return debug;
};
