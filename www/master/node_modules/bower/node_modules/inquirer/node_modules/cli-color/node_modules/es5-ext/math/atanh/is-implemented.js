'use strict';

module.exports = function () {
	var atanh = Math.atanh;
	if (typeof atanh !== 'function') return false;
	return atanh(0.5) === 0.5493061443340549;
};
