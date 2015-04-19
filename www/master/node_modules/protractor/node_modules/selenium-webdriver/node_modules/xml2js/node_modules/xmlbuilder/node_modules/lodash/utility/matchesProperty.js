var baseClone = require('../internal/baseClone'),
    baseMatchesProperty = require('../internal/baseMatchesProperty');

/**
 * Creates a function which compares the property value of `key` on a given
 * object to `value`.
 *
 * **Note:** This method supports comparing arrays, booleans, `Date` objects,
 * numbers, `Object` objects, regexes, and strings. Objects are compared by
 * their own, not inherited, enumerable properties.
 *
 * @static
 * @memberOf _
 * @category Utility
 * @param {string} key The key of the property to get.
 * @param {*} value The value to compare.
 * @returns {Function} Returns the new function.
 * @example
 *
 * var users = [
 *   { 'user': 'barney' },
 *   { 'user': 'fred' },
 *   { 'user': 'pebbles' }
 * ];
 *
 * _.find(users, _.matchesProperty('user', 'fred'));
 * // => { 'user': 'fred', 'age': 40 }
 */
function matchesProperty(key, value) {
  return baseMatchesProperty(key + '', baseClone(value, true));
}

module.exports = matchesProperty;
