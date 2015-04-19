'use strict';

var toPosInt = require('../../number/to-pos-integer')
  , value    = require('../../object/valid-value')

  , hasOwnProperty = Object.prototype.hasOwnProperty;

module.exports = function () {
	var i, l;
	if (!(l = toPosInt(value(this).length))) return null;
	i = l - 1;
	while (!hasOwnProperty.call(this, i)) {
		if (--i === -1) return null;
	}
	return i;
};
