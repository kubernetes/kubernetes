'use strict';
var isFinite = require('is-finite');

module.exports = function (str, n) {
	if (typeof str !== 'string') {
		throw new TypeError('Expected a string as the first argument');
	}

	if (n < 0 || !isFinite(n)) {
		throw new TypeError('Expected a finite positive number');
	}

	var ret = '';

	do {
		if (n & 1) {
			ret += str;
		}

		str += str;
	} while (n = n >> 1);

	return ret;
};
