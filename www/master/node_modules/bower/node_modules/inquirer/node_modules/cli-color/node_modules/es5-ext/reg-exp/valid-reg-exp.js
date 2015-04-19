'use strict';

var isRegExp = require('./is-reg-exp');

module.exports = function (x) {
	if (!isRegExp(x)) throw new TypeError(x + " is not a RegExp object");
	return x;
};
