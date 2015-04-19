var baseEach = require('./baseEach'),
    isLength = require('./isLength');

/**
 * The base implementation of `_.invoke` which requires additional arguments
 * to be provided as an array of arguments rather than individually.
 *
 * @private
 * @param {Array|Object|string} collection The collection to iterate over.
 * @param {Function|string} methodName The name of the method to invoke or
 *  the function invoked per iteration.
 * @param {Array} [args] The arguments to invoke the method with.
 * @returns {Array} Returns the array of results.
 */
function baseInvoke(collection, methodName, args) {
  var index = -1,
      isFunc = typeof methodName == 'function',
      length = collection ? collection.length : 0,
      result = isLength(length) ? Array(length) : [];

  baseEach(collection, function(value) {
    var func = isFunc ? methodName : (value != null && value[methodName]);
    result[++index] = func ? func.apply(value, args) : undefined;
  });
  return result;
}

module.exports = baseInvoke;
