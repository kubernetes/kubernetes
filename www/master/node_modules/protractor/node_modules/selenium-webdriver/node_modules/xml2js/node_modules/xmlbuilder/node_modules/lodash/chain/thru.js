/**
 * This method is like `_.tap` except that it returns the result of `interceptor`.
 *
 * @static
 * @memberOf _
 * @category Chain
 * @param {*} value The value to provide to `interceptor`.
 * @param {Function} interceptor The function to invoke.
 * @param {*} [thisArg] The `this` binding of `interceptor`.
 * @returns {*} Returns the result of `interceptor`.
 * @example
 *
 * _([1, 2, 3])
 *  .last()
 *  .thru(function(value) {
 *    return [value];
 *  })
 *  .value();
 * // => [3]
 */
function thru(value, interceptor, thisArg) {
  return interceptor.call(thisArg, value);
}

module.exports = thru;
