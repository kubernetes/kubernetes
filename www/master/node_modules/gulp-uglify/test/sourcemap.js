'use strict';
var test = require('tape'),
		Vinyl = require('vinyl'),
		gulpUglify = require('../'),
		uglifyjs = require('uglify-js'),
		concat = require('gulp-concat'),
		sourcemaps = require('gulp-sourcemaps');

var testContents1Input = '(function(first, second) {\n    console.log(first + second);\n}(5, 10));\n';
var testContents1Expected = uglifyjs.minify(testContents1Input, {fromString: true}).code;
var testContents2Input = '(function(alert) {\n    alert(5);\n}(alert));\n';
var testContents2Expected = uglifyjs.minify(testContents2Input, {fromString: true}).code;
var testConcatExpected = uglifyjs.minify(testContents1Expected + testContents2Input, {fromString: true}).code;

test('should minify files', function(t) {
	t.plan(11);

	var testFile1 = new Vinyl({
		cwd: "/home/terin/broken-promises/",
		base: "/home/terin/broken-promises/test",
		path: "/home/terin/broken-promises/test/test1.js",
		contents: new Buffer(testContents1Input)
	});

	var sm = sourcemaps.init();
	var mangled = sm.pipe(gulpUglify());

	mangled.on('data', function(newFile) {
		t.ok(newFile, 'emits a file');
		t.ok(newFile.path, 'file has a path');
		t.ok(newFile.relative, 'file has relative path information');
		t.ok(newFile.contents, 'file has contents');

		t.ok(newFile.contents instanceof Buffer, 'file contents are a buffer');

		t.equals(String(newFile.contents), testContents1Expected);

		t.ok(newFile.sourceMap, 'has a source map');
		t.equals(newFile.sourceMap.version, 3, 'source map has expected version');
		t.ok(Array.isArray(newFile.sourceMap.sources), 'source map has sources array');
		t.ok(Array.isArray(newFile.sourceMap.names), 'source maps has names array');
		t.ok(newFile.sourceMap.mappings, 'source map has mappings');
	});

	sm.write(testFile1);
	sm.end();
});

test('should merge source maps correctly', function(t) {
	t.plan(12);

	var testFile1 = new Vinyl({
		cwd: "/home/terin/broken-promises/",
		base: "/home/terin/broken-promises/test",
		path: "/home/terin/broken-promises/test/test1.js",
		contents: new Buffer(testContents1Input)
	});

	var testFile2 = new Vinyl({
		cwd: "/home/terin/broken-promises/",
		base: "/home/terin/broken-promises/test",
		path: "/home/terin/broken-promises/test/test2.js",
		contents: new Buffer(testContents2Input)
	});

	var sm = sourcemaps.init();
	var ct = sm.pipe(concat('all.js'));
	var mangled = ct.pipe(gulpUglify());

	mangled.on('data', function(newFile) {
		t.ok(newFile, 'emits a file');
		t.ok(newFile.path, 'file has a path');
		t.ok(newFile.relative, 'file has relative path information');
		t.ok(newFile.contents, 'file has contents');

		t.ok(newFile.contents instanceof Buffer, 'file contents are a buffer');

		t.equals(String(newFile.contents), testConcatExpected);

		t.ok(newFile.sourceMap, 'has a source map');
		t.equals(newFile.sourceMap.version, 3, 'source map has expected version');
		t.ok(Array.isArray(newFile.sourceMap.sources), 'source map has sources array');
		t.deepEquals(newFile.sourceMap.sources, ['test1.js', 'test2.js'], 'sources array has the inputs');
		t.ok(Array.isArray(newFile.sourceMap.names), 'source maps has names array');
		t.ok(newFile.sourceMap.mappings, 'source map has mappings');
	});

	sm.write(testFile1);
	sm.write(testFile2);
	sm.end();
});

test('should not remember source maps across files', function(t) {
	t.plan(26);

	var testFile1 = new Vinyl({
		cwd: "/home/terin/broken-promises/",
		base: "/home/terin/broken-promises/test",
		path: "/home/terin/broken-promises/test/test1.js",
		contents: new Buffer(testContents1Input)
	});
	testFile1.sourceMap = {
		version: 3,
		file: 'test1.js',
		sourceRoot: '',
		sources: [ 'test1.ts' ],
		sourcesContent: ['(function(first, second) { console.log(first + second) }(5, 10))'],
		names: [],
		mappings: 'AAAA,CAAC,UAAS,KAAK,EAAE,MAAM;IAAI,OAAO,CAAC,GAAG,CAAC,KAAK,GAAG,MAAM,CAAC;AAAC,CAAC,CAAC,CAAC,EAAE,EAAE,CAAC,CAAC'
	};
	var testFile1SourcesContent = [].slice.call(testFile1.sourceMap.sourcesContent);

	var testFile2 = new Vinyl({
		cwd: "/home/terin/broken-promises/",
		base: "/home/terin/broken-promises/test",
		path: "/home/terin/broken-promises/test/test2.js",
		contents: new Buffer(testContents2Input)
	});
	testFile2.sourceMap = {
		version: 3,
		file: 'test2.js',
		sourceRoot: '',
		sources: [ 'test2.ts' ],
		sourcesContent: ['(function(alert) { alert(5); }(alert))'],
		names: [],
		mappings: 'AAAA,CAAC,UAAS,KAAK;IAAI,KAAK,CAAC,CAAC,CAAC;AAAE,CAAC,CAAC,KAAK,CAAC,CAAC'
	};
	var testFile2SourcesContent = [].slice.call(testFile2.sourceMap.sourcesContent);

	var mangled = gulpUglify();

	mangled.on('data', function(newFile) {
		t.ok(newFile, 'emits a file');
		t.ok(newFile.path, 'file has a path');
		t.ok(newFile.relative, 'file has relative path information');
		t.ok(newFile.contents, 'file has contents');

		t.ok(newFile.contents instanceof Buffer, 'file contents are a buffer');

		if (/test1\.js/.test(newFile.path)) {
			t.equals(String(newFile.contents), testContents1Expected);
			t.deepEquals(newFile.sourceMap.sources, ['test1.ts']);
			t.deepEquals(testFile1SourcesContent, newFile.sourceMap.sourcesContent);
		} else if (/test2\.js/.test(newFile.path)) {
			t.equals(String(newFile.contents), testContents2Expected);
			t.deepEquals(newFile.sourceMap.sources, ['test2.ts']);
			t.deepEquals(testFile2SourcesContent, newFile.sourceMap.sourcesContent);
		}

		t.ok(newFile.sourceMap, 'has a source map');
		t.equals(newFile.sourceMap.version, 3, 'source map has expected version');
		t.ok(Array.isArray(newFile.sourceMap.sources), 'source map has sources array');
		t.ok(Array.isArray(newFile.sourceMap.names), 'source maps has names array');
		t.ok(newFile.sourceMap.mappings, 'source map has mappings');
	});

	mangled.write(testFile1);
	mangled.write(testFile2);
	mangled.end();
});
