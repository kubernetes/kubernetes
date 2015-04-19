// Credit: https://github.com/paulmillr/es6-shim/blob/master/es6-shim.js

'use strict';

var toFloat32;

if (typeof Float32Array !== 'undefined') {
	toFloat32 = (function () {
		var float32Array = new Float32Array(1);
		return function (x) {
			float32Array[0] = x;
			return float32Array[0];
		};
	}());
} else {
	toFloat32 = (function () {
		var pack   = require('../_pack-ieee754')
		  , unpack = require('../_unpack-ieee754');

		return function (x) {
			return unpack(pack(x, 8, 23), 8, 23);
		};
	}());
}

module.exports = function (x) {
	if (isNaN(x)) return NaN;
	x = Number(x);
	if (x === 0) return x;
	if (!isFinite(x)) return x;

	return toFloat32(x);
};
