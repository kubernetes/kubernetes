'use strict';
module.exports = Number.isFinite || function (val) {
	// Number.isNaN() => val !== val
	if (typeof val !== 'number' || val !== val || val === Infinity || val === -Infinity) {
		return false;
	}

	return true;
};
