'use strict';

var isPlainObject = require('./is-plain-object')
  , value         = require('./valid-value')

  , keys = Object.keys
  , copy;

copy = function (source) {
	var target = {};
	this[0].push(source);
	this[1].push(target);
	keys(source).forEach(function (key) {
		var index;
		if (!isPlainObject(source[key])) {
			target[key] = source[key];
			return;
		}
		index = this[0].indexOf(source[key]);
		if (index === -1) target[key] = copy.call(this, source[key]);
		else target[key] = this[1][index];
	}, this);
	return target;
};

module.exports = function (source) {
	var obj = Object(value(source));
	if (obj !== source) return obj;
	return copy.call([[], []], obj);
};
