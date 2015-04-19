// Exports true if environment provides native `Symbol` implementation

'use strict';

module.exports = (function () {
	if (typeof Symbol !== 'function') return false;
	return (typeof Symbol.iterator === 'symbol');
}());
