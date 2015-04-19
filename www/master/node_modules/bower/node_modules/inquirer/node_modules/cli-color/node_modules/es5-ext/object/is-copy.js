'use strict';

var eq     = require('./eq')
  , value  = require('./valid-value')

  , keys = Object.keys
  , propertyIsEnumerable = Object.prototype.propertyIsEnumerable;

module.exports = function (a, b) {
	var k1, k2;

	if (eq(value(a), value(b))) return true;

	a = Object(a);
	b = Object(b);

	k1 = keys(a);
	k2 = keys(b);
	if (k1.length !== k2.length) return false;
	return k1.every(function (key) {
		if (!propertyIsEnumerable.call(b, key)) return false;
		return eq(a[key], b[key]);
	});
};
