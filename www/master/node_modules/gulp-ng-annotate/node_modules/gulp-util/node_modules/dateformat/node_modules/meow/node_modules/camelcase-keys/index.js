'use strict';
var mapObj = require('map-obj');
var camelCase = require('camelcase');

module.exports = function (obj) {
	return mapObj(obj, function (key, val) {
		return [camelCase(key), val];
	});
};
