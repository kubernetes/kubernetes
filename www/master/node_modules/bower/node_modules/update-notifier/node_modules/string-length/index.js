'use strict';
var stripAnsi = require('strip-ansi');
var reAstral = /[\uD800-\uDBFF][\uDC00-\uDFFF]/g;

module.exports = function (str) {
	return stripAnsi(str).replace(reAstral, ' ').length;
};
