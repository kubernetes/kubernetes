'use strict';

var isFunction = require('./is-function');

module.exports = function (x) {
	if (!isFunction(x)) throw new TypeError(x + " is not a function");
	return x;
};
