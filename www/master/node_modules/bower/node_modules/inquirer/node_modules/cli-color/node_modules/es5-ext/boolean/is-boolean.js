'use strict';

var toString = Object.prototype.toString

  , id = toString.call(true);

module.exports = function (x) {
	return (typeof x === 'boolean') || ((typeof x === 'object') &&
		((x instanceof Boolean) || (toString.call(x) === id)));
};
