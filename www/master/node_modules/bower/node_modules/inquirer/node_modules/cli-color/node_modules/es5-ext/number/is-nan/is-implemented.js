'use strict';

module.exports = function () {
	var isNaN = Number.isNaN;
	if (typeof isNaN !== 'function') return false;
	return !isNaN({}) && isNaN(NaN) && !isNaN(34);
};
