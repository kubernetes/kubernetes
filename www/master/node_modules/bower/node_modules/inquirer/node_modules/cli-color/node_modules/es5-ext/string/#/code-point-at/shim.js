// Based on: https://github.com/mathiasbynens/String.prototype.codePointAt
// Thanks @mathiasbynens !

'use strict';

var toInteger  = require('../../../number/to-integer')
  , validValue = require('../../../object/valid-value');

module.exports = function (pos) {
	var str = String(validValue(this)), l = str.length, first, second;
	pos = toInteger(pos);

	// Account for out-of-bounds indices:
	if (pos < 0 || pos >= l) return undefined;

	// Get the first code unit
	first = str.charCodeAt(pos);
	if ((first >= 0xD800) && (first <= 0xDBFF) && (l > pos + 1)) {
		second = str.charCodeAt(pos + 1);
		if (second >= 0xDC00 && second <= 0xDFFF) {
			// http://mathiasbynens.be/notes/javascript-encoding#surrogate-formulae
			return (first - 0xD800) * 0x400 + second - 0xDC00 + 0x10000;
		}
	}
	return first;
};
