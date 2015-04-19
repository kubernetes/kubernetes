/** Used as references for `-Infinity` and `Infinity`. */
var POSITIVE_INFINITY = Number.POSITIVE_INFINITY;

/**
 * A specialized version of `_.min` for arrays without support for iteratees.
 *
 * @private
 * @param {Array} array The array to iterate over.
 * @returns {*} Returns the minimum value.
 */
function arrayMin(array) {
  var index = -1,
      length = array.length,
      result = POSITIVE_INFINITY;

  while (++index < length) {
    var value = array[index];
    if (value < result) {
      result = value;
    }
  }
  return result;
}

module.exports = arrayMin;
