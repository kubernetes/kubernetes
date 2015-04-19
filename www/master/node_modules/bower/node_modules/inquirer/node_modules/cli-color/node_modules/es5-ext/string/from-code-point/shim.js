// Based on:
// http://norbertlindenberg.com/2012/05/ecmascript-supplementary-characters/
// and:
// https://github.com/mathiasbynens/String.fromCodePoint/blob/master
// /fromcodepoint.js

'use strict';

var floor = Math.floor, fromCharCode = String.fromCharCode;

module.exports = function (/* …codePoints*/) {
	var chars = [], l = arguments.length, i, c, result = '';
	for (i = 0; i < l; ++i) {
		c = Number(arguments[i]);
		if (!isFinite(c) || c < 0 || c > 0x10FFFF || floor(c) !== c) {
			throw new RangeError("Invalid code point " + c);
		}

		if (c < 0x10000) {
			chars.push(c);
		} else {
			c -= 0x10000;
			chars.push((c >> 10) + 0xD800, (c % 0x400) + 0xDC00);
		}
		if (i + 1 !== l && chars.length <= 0x4000) continue;
		result += fromCharCode.apply(null, chars);
		chars.length = 0;
	}
	return result;
};
