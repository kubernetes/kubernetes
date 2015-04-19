'use strict';
var test = require('tape'),
		Vinyl = require('vinyl'),
		gulpUglify = require('../'),
		Readable = require('stream').Readable,
		Stream = require('stream'),
		PluginError = require('gulp-util/lib/PluginError');
	
var testContentsInput = 'function errorFunction(error) {';

var testFile1 = new Vinyl({
	cwd: "/home/terin/broken-promises/",
	base: "/home/terin/broken-promises/test",
	path: "/home/terin/broken-promises/test/test1.js",
	contents: stringStream()
});

test('should emit error for stream files', function(t) {
	t.plan(6);

	var stream = gulpUglify();

	stream
		.on('data', function() {
			t.fail('should emit error for streams');
		})
		.on('error', function(e) {
			t.pass('emitted error');
			t.ok(e instanceof PluginError, 'error is a PluginError');
			t.equal(e.plugin, 'gulp-uglify', 'error is from gulp-uglify');
			t.equal(e.fileName, testFile1.path, 'error reports the correct file');
			// t.ok(e.stack, 'error has a stack');
			t.skip('error should have a stack');
			t.false(e.showStack, 'error is configured to not print stack');
		});

	stream.write(testFile1);
	stream.end();
});

function stringStream() {
	var stream = new Readable();

	stream._read = function() {
		this.push('terin');
		this.push(null);
	};

	return stream;
}
