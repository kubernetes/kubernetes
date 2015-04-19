'use strict';

var eq            = require('./eq')
  , isPlainObject = require('./is-plain-object')
  , value         = require('./valid-value')

  , isArray = Array.isArray, keys = Object.keys
  , propertyIsEnumerable = Object.prototype.propertyIsEnumerable

  , eqArr, eqVal, eqObj;

eqArr = function (a, b, recMap) {
	var i, l = a.length;
	if (l !== b.length) return false;
	for (i = 0; i < l; ++i) {
		if (a.hasOwnProperty(i) !== b.hasOwnProperty(i)) return false;
		if (!eqVal(a[i], b[i], recMap)) return false;
	}
	return true;
};

eqObj = function (a, b, recMap) {
	var k1 = keys(a), k2 = keys(b);
	if (k1.length !== k2.length) return false;
	return k1.every(function (key) {
		if (!propertyIsEnumerable.call(b, key)) return false;
		return eqVal(a[key], b[key], recMap);
	});
};

eqVal = function (a, b, recMap) {
	var i, eqX, c1, c2;
	if (eq(a, b)) return true;
	if (isPlainObject(a)) {
		if (!isPlainObject(b)) return false;
		eqX = eqObj;
	} else if (isArray(a) && isArray(b)) {
		eqX = eqArr;
	} else {
		return false;
	}
	c1 = recMap[0];
	c2 = recMap[1];
	i = c1.indexOf(a);
	if (i !== -1) {
		if (c2[i].indexOf(b) !== -1) return true;
	} else {
		i = c1.push(a) - 1;
		c2[i] = [];
	}
	c2[i].push(b);
	return eqX(a, b, recMap);
};

module.exports = function (a, b) {
	if (eq(value(a), value(b))) return true;
	return eqVal(Object(a), Object(b), [[], []]);
};
