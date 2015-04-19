'use strict';

module.exports = function (value) {
	value = value >>> 0;
	return value ? 32 - value.toString(2).length : 32;
};
