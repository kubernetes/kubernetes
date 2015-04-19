'use strict';

var exp = Math.exp;

module.exports = function (x) {
	if (isNaN(x)) return NaN;
	x = Number(x);
	if (x === 0) return x;
	if (!isFinite(x)) return x;
	return (exp(x) - exp(-x)) / 2;
};
