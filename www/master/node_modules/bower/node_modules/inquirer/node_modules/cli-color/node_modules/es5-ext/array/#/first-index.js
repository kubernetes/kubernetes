'use strict';

var toPosInt = require('../../number/to-pos-integer')
  , value    = require('../../object/valid-value')

  , hasOwnProperty = Object.prototype.hasOwnProperty;

module.exports = function () {
	var i, l;
	if (!(l = toPosInt(value(this).length))) return null;
	i = 0;
	while (!hasOwnProperty.call(this, i)) {
		if (++i === l) return null;
	}
	return i;
};
