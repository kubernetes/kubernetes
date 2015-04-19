/** Used as the `TypeError` message for "Functions" methods. */
var FUNC_ERROR_TEXT = 'Expected a function';

/**
 * Creates a function that invokes `func` with the `this` binding of the
 * created function and the array of arguments provided to the created
 * function much like [Function#apply](http://es5.github.io/#x15.3.4.3).
 *
 * @static
 * @memberOf _
 * @category Function
 * @param {Function} func The function to spread arguments over.
 * @returns {*} Returns the new function.
 * @example
 *
 * var spread = _.spread(function(who, what) {
 *   return who + ' says ' + what;
 * });
 *
 * spread(['Fred', 'hello']);
 * // => 'Fred says hello'
 *
 * // with a Promise
 * var numbers = Promise.all([
 *   Promise.resolve(40),
 *   Promise.resolve(36)
 * ]);
 *
 * numbers.then(_.spread(function(x, y) {
 *   return x + y;
 * }));
 * // => a Promise of 76
 */
function spread(func) {
  if (typeof func != 'function') {
    throw new TypeError(FUNC_ERROR_TEXT);
  }
  return function(array) {
    return func.apply(this, array);
  };
}

module.exports = spread;
