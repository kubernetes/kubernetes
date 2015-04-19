var isLength = require('./isLength'),
    isObject = require('../lang/isObject'),
    values = require('../object/values');

/**
 * Converts `value` to an array-like object if it is not one.
 *
 * @private
 * @param {*} value The value to process.
 * @returns {Array|Object} Returns the array-like object.
 */
function toIterable(value) {
  if (value == null) {
    return [];
  }
  if (!isLength(value.length)) {
    return values(value);
  }
  return isObject(value) ? value : Object(value);
}

module.exports = toIterable;
