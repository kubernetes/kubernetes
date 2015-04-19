// Credit: https://github.com/paulmillr/es6-shim/

'use strict';

var abs = Math.abs, floor = Math.floor, log = Math.log, min = Math.min
  , pow = Math.pow, LN2 = Math.LN2
  , roundToEven;

roundToEven = function (n) {
	var w = floor(n), f = n - w;
	if (f < 0.5) return w;
	if (f > 0.5) return w + 1;
	return w % 2 ? w + 1 : w;
};

module.exports = function (v, ebits, fbits) {
	var bias = (1 << (ebits - 1)) - 1, s, e, f, i, bits, str, bytes;

	// Compute sign, exponent, fraction
	if (isNaN(v)) {
		// NaN
		// http://dev.w3.org/2006/webapi/WebIDL/#es-type-mapping
		e = (1 << ebits) - 1;
		f = pow(2, fbits - 1);
		s = 0;
	} else if (v === Infinity || v === -Infinity) {
		e = (1 << ebits) - 1;
		f = 0;
		s = (v < 0) ? 1 : 0;
	} else if (v === 0) {
		e = 0;
		f = 0;
		s = (1 / v === -Infinity) ? 1 : 0;
	} else {
		s = v < 0;
		v = abs(v);

		if (v >= pow(2, 1 - bias)) {
			e = min(floor(log(v) / LN2), 1023);
			f = roundToEven(v / pow(2, e) * pow(2, fbits));
			if (f / pow(2, fbits) >= 2) {
				e = e + 1;
				f = 1;
			}
			if (e > bias) {
				// Overflow
				e = (1 << ebits) - 1;
				f = 0;
			} else {
				// Normal
				e = e + bias;
				f = f - pow(2, fbits);
			}
		} else {
			// Subnormal
			e = 0;
			f = roundToEven(v / pow(2, 1 - bias - fbits));
		}
	}

	// Pack sign, exponent, fraction
	bits = [];
	for (i = fbits; i; i -= 1) {
		bits.push(f % 2 ? 1 : 0);
		f = floor(f / 2);
	}
	for (i = ebits; i; i -= 1) {
		bits.push(e % 2 ? 1 : 0);
		e = floor(e / 2);
	}
	bits.push(s ? 1 : 0);
	bits.reverse();
	str = bits.join('');

	// Bits to bytes
	bytes = [];
	while (str.length) {
		bytes.push(parseInt(str.substring(0, 8), 2));
		str = str.substring(8);
	}
	return bytes;
};
