'use strict';

var toArray  = require('./to-array')
  , isDate   = require('../date/is-date')
  , isRegExp = require('../reg-exp/is-reg-exp')

  , isArray = Array.isArray, stringify = JSON.stringify
  , keyValueToString = function (value, key) { return stringify(key) + ':' + exports(value); };

var sparseMap = function (arr) {
	var i, l = arr.length, result = new Array(l);
	for (i = 0; i < l; ++i) {
		if (!arr.hasOwnProperty(i)) continue;
		result[i] = exports(arr[i]);
	}
	return result;
};

module.exports = exports = function (obj) {
	if (obj == null) return String(obj);
	switch (typeof obj) {
	case 'string':
		return stringify(obj);
	case 'number':
	case 'boolean':
	case 'function':
		return String(obj);
	case 'object':
		if (isArray(obj)) return '[' + sparseMap(obj) + ']';
		if (isRegExp(obj)) return String(obj);
		if (isDate(obj)) return 'new Date(' + obj.valueOf() + ')';
		return '{' + toArray(obj, keyValueToString) + '}';
	default:
		throw new TypeError("Serialization of " + String(obj) + "is unsupported");
	}
};
