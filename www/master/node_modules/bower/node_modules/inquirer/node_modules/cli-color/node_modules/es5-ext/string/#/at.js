// Based on: https://github.com/mathiasbynens/String.prototype.at
// Thanks @mathiasbynens !

'use strict';

var toInteger  = require('../../number/to-integer')
  , validValue = require('../../object/valid-value');

module.exports = function (pos) {
	var str = String(validValue(this)), size = str.length
	  , cuFirst, cuSecond, nextPos, len;
	pos = toInteger(pos);

	// Account for out-of-bounds indices
	// The odd lower bound is because the ToInteger operation is
	// going to round `n` to `0` for `-1 < n <= 0`.
	if (pos <= -1 || pos >= size) return '';

	// Second half of `ToInteger`
	pos = pos | 0;
	// Get the first code unit and code unit value
	cuFirst = str.charCodeAt(pos);
	nextPos = pos + 1;
	len = 1;
	if ( // check if it’s the start of a surrogate pair
		(cuFirst >= 0xD800) && (cuFirst <= 0xDBFF) && // high surrogate
			(size > nextPos) // there is a next code unit
	) {
		cuSecond = str.charCodeAt(nextPos);
		if (cuSecond >= 0xDC00 && cuSecond <= 0xDFFF) len = 2; // low surrogate
	}
	return str.slice(pos, pos + len);
};
