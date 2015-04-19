'use strict';

var pow = Math.pow;

module.exports = function (x) {
	if (isNaN(x)) return NaN;
	x = Number(x);
	if (x === 0) return x;
	if (!isFinite(x)) return x;
	if (x < 0) return -pow(-x, 1 / 3);
	return pow(x, 1 / 3);
};
