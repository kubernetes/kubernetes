'use strict';

var ExpectationError = require('./errors').ExpectationError;

module.exports.wrapAssertion = function (func, errorMessage) {
  if (!errorMessage) {
    errorMessage = 'failed assertion' + (func.name ? ' on ' + func.name : '');
  }
  return function (target, callback) {
    var yielded = false;
    var castReturn = function (result) {
      if (yielded) return;
      if (result === true || result === null || result === undefined) {
        callback(null);
      } else if (result === false || typeof result === 'string') {
        callback(new ExpectationError(result || errorMessage));
      } else {
        callback(result);
      }
    };

    try {
      if (func.length >= 2) {
        return func(target, castReturn);
      } else {
        castReturn(func(target));
      }
    } catch (e) {
      castReturn(e);
      return;
    }
  };
};
