'use strict';

var eq   = require('./eq')
  , some = require('./some');

module.exports = function (obj, searchValue) {
	var r;
	return some(obj, function (value, name) {
		if (eq(value, searchValue)) {
			r = name;
			return true;
		}
		return false;
	}) ? r : null;
};
