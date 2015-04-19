
/*
 * Test utils
 */


var body = require('fn-body'),
    normalize = require('normalize-fn');

// given a function, return its body as a normalized string.
// makes tests look a bit cleaner
exports.stringifyFunctionBody = function (fn) {
  return normalize(body(fn));
};
