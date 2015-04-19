var baseAt = require('./baseAt'),
    baseCompareAscending = require('./baseCompareAscending'),
    isIndex = require('./isIndex');

/** Used for native method references. */
var arrayProto = Array.prototype;

/** Native method references. */
var splice = arrayProto.splice;

/**
 * The base implementation of `_.pullAt` without support for individual
 * index arguments.
 *
 * @private
 * @param {Array} array The array to modify.
 * @param {number[]} indexes The indexes of elements to remove.
 * @returns {Array} Returns the new array of removed elements.
 */
function basePullAt(array, indexes) {
  var length = indexes.length,
      result = baseAt(array, indexes);

  indexes.sort(baseCompareAscending);
  while (length--) {
    var index = parseFloat(indexes[length]);
    if (index != previous && isIndex(index)) {
      var previous = index;
      splice.call(array, index, 1);
    }
  }
  return result;
}

module.exports = basePullAt;
