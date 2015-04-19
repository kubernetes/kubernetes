var baseClone = require('../internal/baseClone'),
    baseMatchesProperty = require('../internal/baseMatchesProperty');

/**
 * Creates a function which compares the property value of `path` on a given
 * object to `value`.
 *
 * **Note:** This method supports comparing arrays, booleans, `Date` objects,
 * numbers, `Object` objects, regexes, and strings. Objects are compared by
 * their own, not inherited, enumerable properties.
 *
 * @static
 * @memberOf _
 * @category Utility
 * @param {Array|string} path The path of the property to get.
 * @param {*} value The value to compare.
 * @returns {Function} Returns the new function.
 * @example
 *
 * var users = [
 *   { 'user': 'barney' },
 *   { 'user': 'fred' }
 * ];
 *
 * _.find(users, _.matchesProperty('user', 'fred'));
 * // => { 'user': 'fred' }
 */
function matchesProperty(path, value) {
  return baseMatchesProperty(path, baseClone(value, true));
}

module.exports = matchesProperty;
