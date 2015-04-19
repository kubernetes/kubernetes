'use strict';

var util = require('util');

module.exports.ExpectationError = ExpectationError;

function ExpectationError(message) {
  Error.call(this);
  Error.captureStackTrace(this, this.constructor);

  this.name = this.constructor.name;
  this.message = message;
}
util.inherits(ExpectationError, Error);
