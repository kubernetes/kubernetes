'use strict';

var isArray = Array.isArray;

module.exports = function (t, a) {
	t((t === null) || isArray(t.prototype), true);
};
