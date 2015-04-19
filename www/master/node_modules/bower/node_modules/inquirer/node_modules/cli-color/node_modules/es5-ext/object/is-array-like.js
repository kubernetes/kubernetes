'use strict';

var isFunction = require('../function/is-function')
  , isObject   = require('./is-object');

module.exports = function (x) {
	return ((x != null) && (typeof x.length === 'number') &&

		// Just checking ((typeof x === 'object') && (typeof x !== 'function'))
		// won't work right for some cases, e.g.:
		// type of instance of NodeList in Safari is a 'function'

		((isObject(x) && !isFunction(x)) || (typeof x === "string"))) || false;
};
