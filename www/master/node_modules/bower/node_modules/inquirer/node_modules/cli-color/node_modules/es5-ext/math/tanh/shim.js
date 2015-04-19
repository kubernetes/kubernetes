'use strict';

var exp = Math.exp;

module.exports = function (x) {
	var a, b;
	if (isNaN(x)) return NaN;
	x = Number(x);
	if (x === 0) return x;
	if (x === Infinity) return 1;
	if (x === -Infinity) return -1;
	a = exp(x);
	b = exp(-x);
	return (a - b) / (a + b);
};
