/** Used as the `TypeError` message for "Functions" methods. */
var FUNC_ERROR_TEXT = 'Expected a function';

/**
 * Creates a function to compose other functions into a single function.
 *
 * @private
 * @param {boolean} [fromRight] Specify iterating from right to left.
 * @returns {Function} Returns the new composer function.
 */
function createComposer(fromRight) {
  return function() {
    var length = arguments.length,
        index = length,
        fromIndex = fromRight ? (length - 1) : 0;

    if (!length) {
      return function() { return arguments[0]; };
    }
    var funcs = Array(length);
    while (index--) {
      funcs[index] = arguments[index];
      if (typeof funcs[index] != 'function') {
        throw new TypeError(FUNC_ERROR_TEXT);
      }
    }
    return function() {
      var index = fromIndex,
          result = funcs[index].apply(this, arguments);

      while ((fromRight ? index-- : ++index < length)) {
        result = funcs[index].call(this, result);
      }
      return result;
    };
  };
}

module.exports = createComposer;
