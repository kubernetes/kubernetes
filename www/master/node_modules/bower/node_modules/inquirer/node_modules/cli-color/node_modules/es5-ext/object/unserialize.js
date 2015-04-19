'use strict';

var value  = require('./valid-value');

module.exports = exports = function (code) {
	return (new Function('return ' + value(code)))();
};
