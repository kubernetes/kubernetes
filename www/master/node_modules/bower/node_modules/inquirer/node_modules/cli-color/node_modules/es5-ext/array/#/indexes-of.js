'use strict';

var indexOf = require('./e-index-of');

module.exports = function (value/*, fromIndex*/) {
	var r = [], i, fromIndex = arguments[1];
	while ((i = indexOf.call(this, value, fromIndex)) !== -1) {
		r.push(i);
		fromIndex = i + 1;
	}
	return r;
};
