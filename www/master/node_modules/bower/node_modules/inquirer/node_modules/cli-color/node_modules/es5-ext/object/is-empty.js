'use strict';

var value = require('./valid-value')

  , propertyIsEnumerable = Object.prototype.propertyIsEnumerable;

module.exports = function (obj) {
	var i;
	value(obj);
	for (i in obj) { //jslint: ignore
		if (propertyIsEnumerable.call(obj, i)) return false;
	}
	return true;
};
