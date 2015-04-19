// Taken from: https://github.com/paulmillr/es6-shim/

'use strict';

var toInteger  = require('../../../number/to-integer')
  , toPosInt   = require('../../../number/to-pos-integer')
  , validValue = require('../../../object/valid-value')

  , hasOwnProperty = Object.prototype.hasOwnProperty
  , max = Math.max, min = Math.min;

module.exports = function (target, start/*, end*/) {
	var o = validValue(this), end = arguments[2], l = toPosInt(o.length)
	  , to, from, fin, count, direction;

	target = toInteger(target);
	start = toInteger(start);
	end = (end === undefined) ? l : toInteger(end);

	to = target < 0 ? max(l + target, 0) : min(target, l);
	from = start < 0 ? max(l + start, 0) : min(start, l);
	fin = end < 0 ? max(l + end, 0) : min(end, l);
	count = min(fin - from, l - to);
	direction = 1;

	if ((from < to) && (to < (from + count))) {
		direction = -1;
		from += count - 1;
		to += count - 1;
	}
	while (count > 0) {
		if (hasOwnProperty.call(o, from)) o[to] = o[from];
		else delete o[from];
		from += direction;
		to += direction;
		count -= 1;
	}
	return o;
};
