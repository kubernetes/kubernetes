var LazyWrapper = require('./LazyWrapper'),
    getFuncName = require('./getFuncName'),
    lodash = require('../chain/lodash');

/**
 * Checks if `func` has a lazy counterpart.
 *
 * @private
 * @param {Function} func The function to check.
 * @returns {boolean} Returns `true` if `func` has a lazy counterpart, else `false`.
 */
function isLaziable(func) {
  var funcName = getFuncName(func);
  return !!funcName && func === lodash[funcName] && funcName in LazyWrapper.prototype;
}

module.exports = isLaziable;
