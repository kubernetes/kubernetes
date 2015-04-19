'use strict';

var toPosInt   = require('../../number/to-pos-integer')
  , validValue = require('../../object/valid-value')

  , reduce = Array.prototype.reduce;

module.exports = function (callSite/*,  …substitutions*/) {
	var args, rawValue = Object(validValue(Object(validValue(callSite)).raw));
	if (!toPosInt(rawValue.length)) return '';
	args = arguments;
	return reduce.call(rawValue, function (a, b, i) {
		return a + String(args[i]) + b;
	});
};
