var arrayMap = require('../internal/arrayMap'),
    arrayMax = require('../internal/arrayMax'),
    baseProperty = require('../internal/baseProperty');

/** Used to the length of n-tuples for `_.unzip`. */
var getLength = baseProperty('length');

/**
 * This method is like `_.zip` except that it accepts an array of grouped
 * elements and creates an array regrouping the elements to their pre-`_.zip`
 * configuration.
 *
 * @static
 * @memberOf _
 * @category Array
 * @param {Array} array The array of grouped elements to process.
 * @returns {Array} Returns the new array of regrouped elements.
 * @example
 *
 * var zipped = _.zip(['fred', 'barney'], [30, 40], [true, false]);
 * // => [['fred', 30, true], ['barney', 40, false]]
 *
 * _.unzip(zipped);
 * // => [['fred', 'barney'], [30, 40], [true, false]]
 */
function unzip(array) {
  var index = -1,
      length = (array && array.length && arrayMax(arrayMap(array, getLength))) >>> 0,
      result = Array(length);

  while (++index < length) {
    result[index] = arrayMap(array, baseProperty(index));
  }
  return result;
}

module.exports = unzip;
