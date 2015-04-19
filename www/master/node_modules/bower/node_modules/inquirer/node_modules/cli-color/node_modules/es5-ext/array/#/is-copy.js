'use strict';

var toPosInt = require('../../number/to-pos-integer')
  , eq    = require('../../object/eq')
  , value = require('../../object/valid-value')

  , hasOwnProperty = Object.prototype.hasOwnProperty;

module.exports = function (other) {
	var i, l;
	(value(this) && value(other));
	l = toPosInt(this.length);
	if (l !== toPosInt(other.length)) return false;
	for (i = 0; i < l; ++i) {
		if (hasOwnProperty.call(this, i) !== hasOwnProperty.call(other, i)) {
			return false;
		}
		if (!eq(this[i], other[i])) return false;
	}
	return true;
};
