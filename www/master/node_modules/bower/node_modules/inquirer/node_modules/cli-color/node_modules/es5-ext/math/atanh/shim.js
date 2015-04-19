'use strict';

var log = Math.log;

module.exports = function (x) {
	if (isNaN(x)) return NaN;
	x = Number(x);
	if (x < -1) return NaN;
	if (x > 1) return NaN;
	if (x === -1) return -Infinity;
	if (x === 1) return Infinity;
	if (x === 0) return x;
	return 0.5 * log((1 + x) / (1 - x));
};
