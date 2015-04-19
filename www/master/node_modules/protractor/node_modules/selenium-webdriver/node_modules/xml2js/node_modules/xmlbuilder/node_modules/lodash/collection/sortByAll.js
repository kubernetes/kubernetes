var baseFlatten = require('../internal/baseFlatten'),
    baseSortByOrder = require('../internal/baseSortByOrder'),
    isIterateeCall = require('../internal/isIterateeCall');

/**
 * This method is like `_.sortBy` except that it sorts by property names
 * instead of an iteratee function.
 *
 * @static
 * @memberOf _
 * @category Collection
 * @param {Array|Object|string} collection The collection to iterate over.
 * @param {...(string|string[])} props The property names to sort by,
 *  specified as individual property names or arrays of property names.
 * @returns {Array} Returns the new sorted array.
 * @example
 *
 * var users = [
 *   { 'user': 'barney', 'age': 36 },
 *   { 'user': 'fred',   'age': 40 },
 *   { 'user': 'barney', 'age': 26 },
 *   { 'user': 'fred',   'age': 30 }
 * ];
 *
 * _.map(_.sortByAll(users, ['user', 'age']), _.values);
 * // => [['barney', 26], ['barney', 36], ['fred', 30], ['fred', 40]]
 */
function sortByAll(collection) {
  if (collection == null) {
    return [];
  }
  var args = arguments,
      guard = args[3];

  if (guard && isIterateeCall(args[1], args[2], guard)) {
    args = [collection, args[1]];
  }
  return baseSortByOrder(collection, baseFlatten(args, false, false, 1), []);
}

module.exports = sortByAll;
