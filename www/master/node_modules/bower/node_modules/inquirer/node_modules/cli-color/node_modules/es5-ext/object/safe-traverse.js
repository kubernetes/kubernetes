'use strict';

var value = require('./valid-value');

module.exports = function (obj/*, …names*/) {
	var length, current = 1;
	value(obj);
	length = arguments.length - 1;
	if (!length) return obj;
	while (current < length) {
		obj = obj[arguments[current++]];
		if (obj == null) return undefined;
	}
	return obj[arguments[current]];
};
