'use strict';

var log = Math.log, LOG2E = Math.LOG2E;

module.exports = function (x) {
	if (isNaN(x)) return NaN;
	x = Number(x);
	if (x < 0) return NaN;
	if (x === 0) return -Infinity;
	if (x === 1) return 0;
	if (x === Infinity) return Infinity;

	return log(x) * LOG2E;
};
