/**
 * The inverse of `_.property`; this method creates a function which returns
 * the property value of a given key on `object`.
 *
 * @static
 * @memberOf _
 * @category Utility
 * @param {Object} object The object to inspect.
 * @returns {Function} Returns the new function.
 * @example
 *
 * var object = { 'a': 3, 'b': 1, 'c': 2 };
 *
 * _.map(['a', 'c'], _.propertyOf(object));
 * // => [3, 2]
 *
 * _.sortBy(['a', 'b', 'c'], _.propertyOf(object));
 * // => ['b', 'c', 'a']
 */
function propertyOf(object) {
  return function(key) {
    return object == null ? undefined : object[key];
  };
}

module.exports = propertyOf;
