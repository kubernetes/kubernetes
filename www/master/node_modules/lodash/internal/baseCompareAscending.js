/**
 * The base implementation of `compareAscending` which compares values and
 * sorts them in ascending order without guaranteeing a stable sort.
 *
 * @private
 * @param {*} value The value to compare to `other`.
 * @param {*} other The value to compare to `value`.
 * @returns {number} Returns the sort order indicator for `value`.
 */
function baseCompareAscending(value, other) {
  if (value !== other) {
    var valIsReflexive = value === value,
        othIsReflexive = other === other;

    if (value > other || !valIsReflexive || (value === undefined && othIsReflexive)) {
      return 1;
    }
    if (value < other || !othIsReflexive || (other === undefined && valIsReflexive)) {
      return -1;
    }
  }
  return 0;
}

module.exports = baseCompareAscending;
