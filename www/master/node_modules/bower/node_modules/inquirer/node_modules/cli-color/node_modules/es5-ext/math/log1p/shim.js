// Thanks: https://github.com/monolithed/ECMAScript-6/blob/master/ES6.js

'use strict';

var log = Math.log;

module.exports = function (x) {
	if (isNaN(x)) return NaN;
	x = Number(x);
	if (x < -1) return NaN;
	if (x === -1) return -Infinity;
	if (x === 0) return x;
	if (x === Infinity) return Infinity;

	if (x > -1.0e-8 && x < 1.0e-8) return (x - x * x / 2);
	return log(1 + x);
};
