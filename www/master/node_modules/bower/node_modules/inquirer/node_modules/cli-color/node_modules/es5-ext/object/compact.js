'use strict';

var filter = require('./filter');

module.exports = function (obj) {
	return filter(obj, function (val) { return val != null; });
};
