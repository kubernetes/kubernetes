'use strict';

var strCompare = require('../string/#/case-insensitive-compare')
  , isObject   = require('./is-object')

  , resolve, typeMap;

typeMap = {
	undefined: 0,
	object: 1,
	boolean: 2,
	string: 3,
	number: 4
};

resolve = function (a) {
	if (isObject(a)) {
		if (typeof a.valueOf !== 'function') return NaN;
		a = a.valueOf();
		if (isObject(a)) {
			if (typeof a.toString !== 'function') return NaN;
			a = a.toString();
			if (typeof a !== 'string') return NaN;
		}
	}
	return a;
};

module.exports = function (a, b) {
	if (a === b) return 0; // Same

	a = resolve(a);
	b = resolve(b);
	if (a == b) return typeMap[typeof a] - typeMap[typeof b]; //jslint: ignore
	if (a == null) return -1;
	if (b == null) return 1;
	if ((typeof a === 'string') || (typeof b === 'string')) {
		return strCompare.call(a, b);
	}
	if ((a !== a) && (b !== b)) return 0; //jslint: ignore
	return Number(a) - Number(b);
};
