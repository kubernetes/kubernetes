var arrayReduceRight = require('../internal/arrayReduceRight'),
    baseCallback = require('../internal/baseCallback'),
    baseEachRight = require('../internal/baseEachRight'),
    baseReduce = require('../internal/baseReduce'),
    isArray = require('../lang/isArray');

/**
 * This method is like `_.reduce` except that it iterates over elements of
 * `collection` from right to left.
 *
 * @static
 * @memberOf _
 * @alias foldr
 * @category Collection
 * @param {Array|Object|string} collection The collection to iterate over.
 * @param {Function} [iteratee=_.identity] The function invoked per iteration.
 * @param {*} [accumulator] The initial value.
 * @param {*} [thisArg] The `this` binding of `iteratee`.
 * @returns {*} Returns the accumulated value.
 * @example
 *
 * var array = [[0, 1], [2, 3], [4, 5]];
 *
 * _.reduceRight(array, function(flattened, other) {
 *   return flattened.concat(other);
 * }, []);
 * // => [4, 5, 2, 3, 0, 1]
 */
function reduceRight(collection, iteratee, accumulator, thisArg) {
  var func = isArray(collection) ? arrayReduceRight : baseReduce;
  return func(collection, baseCallback(iteratee, thisArg, 4), accumulator, arguments.length < 3, baseEachRight);
}

module.exports = reduceRight;
