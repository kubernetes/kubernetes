'use strict';

var log = Math.log, sqrt = Math.sqrt;

module.exports = function (x) {
	if (isNaN(x)) return NaN;
	x = Number(x);
	if (x < 1) return NaN;
	if (x === 1) return 0;
	if (x === Infinity) return x;
	return log(x + sqrt(x * x - 1));
};
