'use strict';

module.exports = function (x, y) {
	return ((x === y) || ((x !== x) && (y !== y))); //jslint: ignore
};
