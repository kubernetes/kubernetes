'use strict';

var isError = require('./is-error');

module.exports = function (x) {
	if (!isError(x)) throw new TypeError(x + " is not an Error object");
	return x;
};
