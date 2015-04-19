'use strict';
var gutil = require('gulp-util');
var through = require('through2');
var multimatch = require('multimatch');

module.exports = function (pattern, options) {
	pattern = typeof pattern === 'string' ? [pattern] : pattern;
	options = options || {};

	if (!Array.isArray(pattern) && typeof pattern !== 'function') {
		throw new gutil.PluginError('gulp-filter', '`pattern` should be a string, array, or function');
	}

	var restoreStream = through.obj();

	var stream = through.obj(function (file, enc, cb) {
		var match = typeof pattern === 'function' ? pattern(file) :
					multimatch(file.relative, pattern, options).length > 0;

		if (match) {
			cb(null, file);
			return;
		}

		restoreStream.write(file);
	  	cb();
	}, function (cb) {
		restoreStream.end();
		cb();
	});

	stream.restore = function (options) {
		options = options || {};

		if (options.end) {
			return restoreStream;
		}

		return restoreStream.pipe(through.obj(), {end: false});
	};

	return stream;
};
