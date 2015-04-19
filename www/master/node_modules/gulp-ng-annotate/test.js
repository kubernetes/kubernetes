"use strict";

var Readable = require("stream").Readable;
var assert = require("assert");
var gutil = require("gulp-util");
var ngAnnotate = require("./index");
var sourcemaps = require("gulp-sourcemaps");

var ORIGINAL = 'angular.module("test"); m.directive("foo", function($a, $b) {});';
var TRANSFORMED = 'angular.module("test"); m.directive("foo", ["$a", "$b", function($a, $b) {}]);';
var BAD_INPUT = 'angular.module("test").directive("foo", function$a, $b) {});';

describe("gulp-ng-annotate", function() {
  it("should annotate angular declarations", function (done) {
    var stream = ngAnnotate();

    stream.on("data", function (data) {
      assert.equal(data.contents.toString(), TRANSFORMED);
      done();
    });

    stream.write(new gutil.File({contents: new Buffer(ORIGINAL)}));
  });

  it("should not touch already annotated declarations", function (done) {
    var stream = ngAnnotate();

    stream.on("data", function (data) {
      assert.equal(data.contents.toString(), TRANSFORMED);
      done();
    });

    stream.write(new gutil.File({contents: new Buffer(TRANSFORMED)}));
  });

  it("should emit PluginError on bad input", function (done) {
    var stream = ngAnnotate();

    try {
      stream.write(new gutil.File({contents: new Buffer(BAD_INPUT)}));
    } catch (err) {
      assert(err instanceof gutil.PluginError);
      assert.equal(err.message.slice(0, 7), "error: ")
      done();
    }
  });

  it("should support passing ng-annotate options", function (done) {
    var stream = ngAnnotate({remove: true});

    stream.on("data", function (data) {
      assert.equal(data.contents.toString(), ORIGINAL);
      done();
    });

    stream.write(new gutil.File({contents: new Buffer(TRANSFORMED)}));
  });

  it("should show filename on error", function (done) {
    var stream = ngAnnotate();

    try {
      stream.write(new gutil.File({path: "1.js", contents: new Buffer(BAD_INPUT)}));
    } catch (err) {
      assert(err instanceof gutil.PluginError);
      assert.equal(err.message.slice(0, 13), "1.js: error: ")
      done();
    }
  });

  it("should support source maps", function (done) {
    var stream = sourcemaps.init()
    stream.write(new gutil.File({path: "1.js", contents: new Buffer(ORIGINAL)}));
    stream.pipe(ngAnnotate()).on("data", function (data) {
      assert.equal(data.contents.toString(), TRANSFORMED);
      assert.deepEqual(data.sourceMap.sourcesContent, [ORIGINAL]);
      assert.deepEqual(data.sourceMap.sources, ["1.js"]);
      done();
    });
  });

  it("should allow to skip source map generation", function (done) {
    var stream = sourcemaps.init()
    stream.write(new gutil.File({path: "1.js", contents: new Buffer(ORIGINAL)}));
    stream.pipe(ngAnnotate({sourcemap: false})).on("data", function (data) {
      assert.equal(data.sourceMap.mappings, "");
      done();
    });
  });

  it("should preserve file attribute in the sourcemap object", function (done) {
    var stream = sourcemaps.init()
    stream.write(new gutil.File({path: "1.js", contents: new Buffer(ORIGINAL)}));
    stream.pipe(ngAnnotate()).on("data", function (data) {
      assert.equal(data.sourceMap.file, "1.js");
      done();
    });
  });

  it("should support streams", function(done) {
    var stream = ngAnnotate();
    var contentsStream = new Readable();
    contentsStream.push(ORIGINAL);
    contentsStream.push(null);

    stream.on("data", function (file) {
      file.contents.on("data", function(data) {
        assert.equal(data, TRANSFORMED);
        done();
      });
    });

    stream.write(new gutil.File({contents: contentsStream}));
  });
});
