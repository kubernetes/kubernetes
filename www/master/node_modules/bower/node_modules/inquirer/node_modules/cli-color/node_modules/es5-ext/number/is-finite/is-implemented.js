'use strict';

module.exports = function () {
	var isFinite = Number.isFinite;
	if (typeof isFinite !== 'function') return false;
	return !isFinite('23') && isFinite(34) && !isFinite(Infinity);
};
