'use strict';

var value   = require('../object/valid-value');

module.exports = function (name) {
	return function (o) { return value(o)[name]; };
};
