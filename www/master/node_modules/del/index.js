'use strict';
var path = require('path');
var globby = require('globby');
var eachAsync = require('each-async');
var isPathCwd = require('is-path-cwd');
var isPathInCwd = require('is-path-in-cwd');
var rimraf = require('rimraf');
var objectAssign = require('object-assign');

function safeCheck(file) {
	if (isPathCwd(file)) {
		throw new Error('Cannot delete the current working directory. Can be overriden with the `force` option.');
	}

	if (!isPathInCwd(file)) {
		throw new Error('Cannot delete files/folders outside the current working directory. Can be overriden with the `force` option.');
	}
}

module.exports = function (patterns, opts, cb) {
	if (typeof opts !== 'object') {
		cb = opts;
		opts = {};
	}

	opts = objectAssign({}, opts);
	cb = cb || function () {};

	var force = opts.force;
	delete opts.force;

	var deletedFiles = [];

	globby(patterns, opts, function (err, files) {
		if (err) {
			cb(err);
			return;
		}

		eachAsync(files, function (el, i, next) {
			if (!force) {
				safeCheck(el);
			}

			el = path.resolve(opts.cwd || '', el);
			deletedFiles.push(el);
			rimraf(el, next);
		}, function (err) {
			if (err) {
				cb(err);
				return;
			}

			cb(null, deletedFiles);
		});
	});
};

module.exports.sync = function (patterns, opts) {
	opts = objectAssign({}, opts);

	var force = opts.force;
	delete opts.force;

	var deletedFiles = [];

	globby.sync(patterns, opts).forEach(function (el) {
		if (!force) {
			safeCheck(el);
		}

		el = path.resolve(opts.cwd || '', el);
		deletedFiles.push(el);
		rimraf.sync(el);
	});

	return deletedFiles;
};
