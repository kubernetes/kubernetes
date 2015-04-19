var baseCallback = require('../internal/baseCallback'),
    baseEachRight = require('../internal/baseEachRight'),
    baseFind = require('../internal/baseFind');

/**
 * This method is like `_.find` except that it iterates over elements of
 * `collection` from right to left.
 *
 * @static
 * @memberOf _
 * @category Collection
 * @param {Array|Object|string} collection The collection to search.
 * @param {Function|Object|string} [predicate=_.identity] The function invoked
 *  per iteration.
 * @param {*} [thisArg] The `this` binding of `predicate`.
 * @returns {*} Returns the matched element, else `undefined`.
 * @example
 *
 * _.findLast([1, 2, 3, 4], function(n) {
 *   return n % 2 == 1;
 * });
 * // => 3
 */
function findLast(collection, predicate, thisArg) {
  predicate = baseCallback(predicate, thisArg, 3);
  return baseFind(collection, predicate, baseEachRight);
}

module.exports = findLast;
