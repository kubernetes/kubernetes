// Thanks: https://github.com/monolithed/ECMAScript-6

'use strict';

var exp = Math.exp;

module.exports = function (x) {
	if (isNaN(x)) return NaN;
	x = Number(x);
	if (x === 0) return x;
	if (x === Infinity) return Infinity;
	if (x === -Infinity) return -1;

	if ((x > -1.0e-6) && (x < 1.0e-6)) return x + x * x / 2;
	return exp(x) - 1;
};
