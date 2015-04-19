'use strict';

var toString = Object.prototype.toString

  , id = toString.call(1);

module.exports = function (x) {
	return ((typeof x === 'number') ||
		((x instanceof Number) ||
			((typeof x === 'object') && (toString.call(x) === id))));
};
