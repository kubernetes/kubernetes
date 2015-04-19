var baseAt = require('../internal/baseAt'),
    baseFlatten = require('../internal/baseFlatten'),
    isLength = require('../internal/isLength'),
    toIterable = require('../internal/toIterable');

/**
 * Creates an array of elements corresponding to the given keys, or indexes,
 * of `collection`. Keys may be specified as individual arguments or as arrays
 * of keys.
 *
 * @static
 * @memberOf _
 * @category Collection
 * @param {Array|Object|string} collection The collection to iterate over.
 * @param {...(number|number[]|string|string[])} [props] The property names
 *  or indexes of elements to pick, specified individually or in arrays.
 * @returns {Array} Returns the new array of picked elements.
 * @example
 *
 * _.at(['a', 'b', 'c'], [0, 2]);
 * // => ['a', 'c']
 *
 * _.at(['fred', 'barney', 'pebbles'], 0, 2);
 * // => ['fred', 'pebbles']
 */
function at(collection) {
  var length = collection ? collection.length : 0;
  if (isLength(length)) {
    collection = toIterable(collection);
  }
  return baseAt(collection, baseFlatten(arguments, false, false, 1));
}

module.exports = at;
