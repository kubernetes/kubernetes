'use strict';

var setPrototypeOf    = require('es5-ext/object/set-prototype-of')
  , object            = require('es5-ext/object/valid-object')
  , value             = require('es5-ext/object/valid-value')
  , d                 = require('d')
  , getIterator       = require('es6-iterator/get')
  , forOf             = require('es6-iterator/for-of')
  , toStringTagSymbol = require('es6-symbol').toStringTag
  , isNative          = require('./is-native-implemented')

  , isArray = Array.isArray, defineProperty = Object.defineProperty, random = Math.random
  , hasOwnProperty = Object.prototype.hasOwnProperty
  , genId, WeakMapPoly;

genId = (function () {
	var generated = Object.create(null);
	return function () {
		var id;
		do { id = random().toString(36).slice(2); } while (generated[id]);
		generated[id] = true;
		return id;
	};
}());

module.exports = WeakMapPoly = function (/*iterable*/) {
	var iterable = arguments[0];
	if (!(this instanceof WeakMapPoly)) return new WeakMapPoly(iterable);
	if (this.__weakMapData__ !== undefined) {
		throw new TypeError(this + " cannot be reinitialized");
	}
	if (iterable != null) {
		if (!isArray(iterable)) iterable = getIterator(iterable);
	}
	defineProperty(this, '__weakMapData__', d('c', '$weakMap$' + genId()));
	if (!iterable) return;
	forOf(iterable, function (val) {
		value(val);
		this.set(val[0], val[1]);
	}, this);
};

if (isNative) {
	if (setPrototypeOf) setPrototypeOf(WeakMapPoly, WeakMap);
	WeakMapPoly.prototype = Object.create(WeakMap.prototype, {
		constructor: d(WeakMapPoly)
	});
}

Object.defineProperties(WeakMapPoly.prototype, {
	clear: d(function () {
		defineProperty(this, '__weakMapData__', d('c', '$weakMap$' + genId()));
	}),
	delete: d(function (key) {
		if (hasOwnProperty.call(object(key), this.__weakMapData__)) {
			delete key[this.__weakMapData__];
			return true;
		}
		return false;
	}),
	get: d(function (key) {
		if (hasOwnProperty.call(object(key), this.__weakMapData__)) {
			return key[this.__weakMapData__];
		}
	}),
	has: d(function (key) {
		return hasOwnProperty.call(object(key), this.__weakMapData__);
	}),
	set: d(function (key, value) {
		defineProperty(object(key), this.__weakMapData__, d('c', value));
		return this;
	}),
	toString: d(function () { return '[object WeakMap]'; })
});
defineProperty(WeakMapPoly.prototype, toStringTagSymbol, d('c', 'WeakMap'));
