'use strict';
var fs = require('fs');
var path = require('path');
var crypto = require('crypto');
var gutil = require('gulp-util');
var through = require('through2');

// ignore missing file error
function fsOperationFailed(stream, sourceFile, err) {
	if (err) {
		if (err.code !== 'ENOENT') {
			stream.emit('error', new gutil.PluginError('gulp-changed', err, {
				fileName: sourceFile.path
			}));
		}

		stream.push(sourceFile);
	}

	return err;
}

function sha1(buf) {
	return crypto.createHash('sha1').update(buf).digest('hex');
}

// only push through files changed more recently than the destination files
function compareLastModifiedTime(stream, cb, sourceFile, targetPath) {
	fs.stat(targetPath, function (err, targetStat) {
		if (!fsOperationFailed(stream, sourceFile, err)) {
			if (sourceFile.stat.mtime > targetStat.mtime) {
				stream.push(sourceFile);
			}
		}

		cb();
	});
}

// only push through files with different SHA1 than the destination files
function compareSha1Digest(stream, cb, sourceFile, targetPath) {
	fs.readFile(targetPath, function (err, targetData) {
		if (sourceFile.isNull()) {
			cb(null, sourceFile);
			return;
		}

		if (!fsOperationFailed(stream, sourceFile, err)) {
			var sourceDigest = sha1(sourceFile.contents);
			var targetDigest = sha1(targetData);

			if (sourceDigest !== targetDigest) {
				stream.push(sourceFile);
			}
		}

		cb();
	});
}

module.exports = function (dest, opts) {
	opts = opts || {};
	opts.cwd = opts.cwd || process.cwd();
	opts.hasChanged = opts.hasChanged || compareLastModifiedTime;

	if (!dest) {
		throw new gutil.PluginError('gulp-changed', '`dest` required');
	}

	return through.obj(function (file, enc, cb) {
		var dest2 = typeof dest === 'function' ? dest(file) : dest;
		var newPath = path.resolve(opts.cwd, dest2, file.relative);

		if (opts.extension) {
			newPath = gutil.replaceExtension(newPath, opts.extension);
		}

		opts.hasChanged(this, cb, file, newPath);
	});
};

module.exports.compareLastModifiedTime = compareLastModifiedTime;
module.exports.compareSha1Digest = compareSha1Digest;
