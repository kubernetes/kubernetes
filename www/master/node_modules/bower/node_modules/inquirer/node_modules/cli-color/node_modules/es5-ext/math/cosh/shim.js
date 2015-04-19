'use strict';

var exp = Math.exp;

module.exports = function (x) {
	if (isNaN(x)) return NaN;
	x = Number(x);
	if (x === 0) return 1;
	if (!isFinite(x)) return Infinity;
	return (exp(x) + exp(-x)) / 2;
};
