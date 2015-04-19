var baseCallback = require('./baseCallback'),
    charAtCallback = require('./charAtCallback'),
    extremumBy = require('./extremumBy'),
    isArray = require('../lang/isArray'),
    isIterateeCall = require('./isIterateeCall'),
    isString = require('../lang/isString'),
    toIterable = require('./toIterable');

/**
 * Creates a `_.max` or `_.min` function.
 *
 * @private
 * @param {Function} arrayFunc The function to get the extremum value from an array.
 * @param {boolean} [isMin] Specify returning the minimum, instead of the maximum,
 *  extremum value.
 * @returns {Function} Returns the new extremum function.
 */
function createExtremum(arrayFunc, isMin) {
  return function(collection, iteratee, thisArg) {
    if (thisArg && isIterateeCall(collection, iteratee, thisArg)) {
      iteratee = null;
    }
    var noIteratee = iteratee == null;

    iteratee = noIteratee ? iteratee : baseCallback(iteratee, thisArg, 3);
    if (noIteratee) {
      var isArr = isArray(collection);
      if (!isArr && isString(collection)) {
        iteratee = charAtCallback;
      } else {
        return arrayFunc(isArr ? collection : toIterable(collection));
      }
    }
    return extremumBy(collection, iteratee, isMin);
  };
}

module.exports = createExtremum;
