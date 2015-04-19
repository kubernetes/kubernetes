'use strict';
var isUtf8 = require('is-utf8');

var stripBom = module.exports = function (arg) {
	if (typeof arg === 'string') {
		return arg.replace(/^\ufeff/g, '');
	}

	if (Buffer.isBuffer(arg) && isUtf8(arg) &&
		arg[0] === 0xef && arg[1] === 0xbb && arg[2] === 0xbf) {
		return arg.slice(3);
	}

	return arg;
};

stripBom.stream = function () {
	var firstChunk = require('first-chunk-stream');

	return firstChunk({minSize: 3}, function (chunk, enc, cb) {
		this.push(stripBom(chunk));
		cb();
	});
};
