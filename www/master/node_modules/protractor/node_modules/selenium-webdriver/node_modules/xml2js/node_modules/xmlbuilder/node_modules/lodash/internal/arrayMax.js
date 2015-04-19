/** Used as references for `-Infinity` and `Infinity`. */
var NEGATIVE_INFINITY = Number.NEGATIVE_INFINITY;

/**
 * A specialized version of `_.max` for arrays without support for iteratees.
 *
 * @private
 * @param {Array} array The array to iterate over.
 * @returns {*} Returns the maximum value.
 */
function arrayMax(array) {
  var index = -1,
      length = array.length,
      result = NEGATIVE_INFINITY;

  while (++index < length) {
    var value = array[index];
    if (value > result) {
      result = value;
    }
  }
  return result;
}

module.exports = arrayMax;
