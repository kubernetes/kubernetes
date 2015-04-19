'use strict';

var log = Math.log, sqrt = Math.sqrt;

module.exports = function (x) {
	if (isNaN(x)) return NaN;
	x = Number(x);
	if (x === 0) return x;
	if (!isFinite(x)) return x;
	if (x < 0) {
		x = -x;
		return -log(x + sqrt(x * x + 1));
	}
	return log(x + sqrt(x * x + 1));
};
