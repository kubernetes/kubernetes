// Exports true if environment provides native `WeakMap` implementation,
// whatever that is.

'use strict';

module.exports = (function () {
	if (typeof WeakMap === 'undefined') return false;
	return (Object.prototype.toString.call(WeakMap.prototype) ===
			'[object WeakMap]');
}());
