// Thanks: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference
//         /Global_Objects/Math/imul

'use strict';

module.exports = function (x, y) {
	var xh  = (x >>> 16) & 0xffff, xl = x & 0xffff
	  , yh  = (y >>> 16) & 0xffff, yl = y & 0xffff;

	// the shift by 0 fixes the sign on the high part
	// the final |0 converts the unsigned value into a signed value
	return ((xl * yl) + (((xh * yl + xl * yh) << 16) >>> 0) | 0);
};
