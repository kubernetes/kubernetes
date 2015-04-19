'use strict';

var isDate = require('./is-date');

module.exports = function (x) {
	if (!isDate(x)) throw new TypeError(x + " is not a Date object");
	return x;
};
