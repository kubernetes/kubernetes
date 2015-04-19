var baseEach = require('./baseEach');

/** Used as references for `-Infinity` and `Infinity`. */
var NEGATIVE_INFINITY = Number.NEGATIVE_INFINITY,
    POSITIVE_INFINITY = Number.POSITIVE_INFINITY;

/**
 * Gets the extremum value of `collection` invoking `iteratee` for each value
 * in `collection` to generate the criterion by which the value is ranked.
 * The `iteratee` is invoked with three arguments; (value, index, collection).
 *
 * @private
 * @param {Array|Object|string} collection The collection to iterate over.
 * @param {Function} iteratee The function invoked per iteration.
 * @param {boolean} [isMin] Specify returning the minimum, instead of the
 *  maximum, extremum value.
 * @returns {*} Returns the extremum value.
 */
function extremumBy(collection, iteratee, isMin) {
  var exValue = isMin ? POSITIVE_INFINITY : NEGATIVE_INFINITY,
      computed = exValue,
      result = computed;

  baseEach(collection, function(value, index, collection) {
    var current = iteratee(value, index, collection);
    if ((isMin ? (current < computed) : (current > computed)) ||
        (current === exValue && current === result)) {
      computed = current;
      result = value;
    }
  });
  return result;
}

module.exports = extremumBy;
