var arrayEachRight = require('../internal/arrayEachRight'),
    baseEachRight = require('../internal/baseEachRight'),
    bindCallback = require('../internal/bindCallback'),
    isArray = require('../lang/isArray');

/**
 * This method is like `_.forEach` except that it iterates over elements of
 * `collection` from right to left.
 *
 * @static
 * @memberOf _
 * @alias eachRight
 * @category Collection
 * @param {Array|Object|string} collection The collection to iterate over.
 * @param {Function} [iteratee=_.identity] The function invoked per iteration.
 * @param {*} [thisArg] The `this` binding of `iteratee`.
 * @returns {Array|Object|string} Returns `collection`.
 * @example
 *
 * _([1, 2]).forEachRight(function(n) {
 *   console.log(n);
 * }).join(',');
 * // => logs each value from right to left and returns the array
 */
function forEachRight(collection, iteratee, thisArg) {
  return (typeof iteratee == 'function' && typeof thisArg == 'undefined' && isArray(collection))
    ? arrayEachRight(collection, iteratee)
    : baseEachRight(collection, bindCallback(iteratee, thisArg, 3));
}

module.exports = forEachRight;
