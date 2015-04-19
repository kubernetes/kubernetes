// Credit: https://github.com/paulmillr/es6-shim/

'use strict';

var pow = Math.pow;

module.exports = function (bytes, ebits, fbits) {
	// Bytes to bits
	var bits = [], i, j, b, str,
	bias, s, e, f;

	for (i = bytes.length; i; i -= 1) {
		b = bytes[i - 1];
		for (j = 8; j; j -= 1) {
			bits.push(b % 2 ? 1 : 0);
			b = b >> 1;
		}
	}
	bits.reverse();
	str = bits.join('');

	// Unpack sign, exponent, fraction
	bias = (1 << (ebits - 1)) - 1;
	s = parseInt(str.substring(0, 1), 2) ? -1 : 1;
	e = parseInt(str.substring(1, 1 + ebits), 2);
	f = parseInt(str.substring(1 + ebits), 2);

	// Produce number
	if (e === (1 << ebits) - 1) return f !== 0 ? NaN : s * Infinity;
	if (e > 0) return s * pow(2, e - bias) * (1 + f / pow(2, fbits));
	if (f !== 0) return s * pow(2, -(bias - 1)) * (f / pow(2, fbits));
	return s < 0 ? -0 : 0;
};
