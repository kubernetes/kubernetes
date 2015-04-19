"use strict";

var through = require('through2');
var gulpmatch = require('gulp-match');

var include = function(condition){
	return through.obj(function (file, enc, callback) {
		if (gulpmatch(file,condition)) {
			this.push(file);
		}
		return callback();
	});
};

var exclude = function(condition){
	return through.obj(function (file, enc, callback) {
		if (!gulpmatch(file,condition)) {
			this.push(file);
		}
		return callback();
	});
};

module.exports = exclude;
module.exports.include = include;
module.exports.exclude = exclude;
