'use strict';
var test = require('tape'),
		Vinyl = require('vinyl'),
		gulpUglify = require('../');
	

test('should preserve all comments', function(t) {
	t.plan(3);

	var testFile1 = new Vinyl({
		cwd: "/home/terin/broken-promises/",
		base: "/home/terin/broken-promises/test",
		path: "/home/terin/broken-promises/test/test1.js",
		contents: new Buffer('/* comment one *//*! comment two *//* comment three */')
	});

	var stream = gulpUglify({ preserveComments: 'all' });

	stream.on('data', function(newFile) {
		var contents = newFile.contents.toString();
		t.ok(/one/.test(contents), 'has comment one');
		t.ok(/two/.test(contents), 'has comment two');
		t.ok(/three/.test(contents), 'has comment three');
	});

	stream.write(testFile1);
	stream.end();
});

test('should preserve important comments', function(t) {
	t.plan(3);

	var testFile1 = new Vinyl({
		cwd: "/home/terin/broken-promises/",
		base: "/home/terin/broken-promises/test",
		path: "/home/terin/broken-promises/test/test1.js",
		contents: new Buffer('/* comment one *//*! comment two *//* comment three */')
	});

	var stream = gulpUglify({ preserveComments: 'some' });

	stream.on('data', function(newFile) {
		var contents = newFile.contents.toString();
		t.false(/one/.test(contents), 'does not have comment one');
		t.ok(/two/.test(contents), 'has comment two');
		t.false(/three/.test(contents), 'does not have comment three');
	});

	stream.write(testFile1);
	stream.end();
});

test('should preserve comments that fn returns true for', function(t) {
	t.plan(3);

	var testFile1 = new Vinyl({
		cwd: "/home/terin/broken-promises/",
		base: "/home/terin/broken-promises/test",
		path: "/home/terin/broken-promises/test/test1.js",
		contents: new Buffer('/* comment one *//*! comment two *//* comment three */')
	});

	var stream = gulpUglify({
		preserveComments: function(node, comment) {
			return /three/.test(comment.value);
		}
	});

	stream.on('data', function(newFile) {
		var contents = newFile.contents.toString();
		t.false(/one/.test(contents), 'does not have comment one');
		t.false(/two/.test(contents), 'does not have comment two');
		t.true(/three/.test(contents), 'has comment three');
	});

	stream.write(testFile1);
	stream.end();
});
