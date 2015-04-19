/*
 * examples.js: Macros for testing code in examples/
 *
 * (C) 2010 Nodejitsu Inc.
 * MIT LICENCE
 *
 */

var assert = require('assert'),
    fs = require('fs'),
    path = require('path'),
    spawn = require('child_process').spawn,
    async = require('async');

var rootDir = path.join(__dirname, '..', '..'),
    examplesDir = path.join(rootDir, 'examples');

//
// ### function shouldHaveDeps ()
//
// Ensures that all `npm` dependencies are installed in `/examples`.
//
exports.shouldHaveDeps = function () {
  return {
    "Before testing examples": {
      topic: function () {
        async.waterfall([
          //
          // 1. Read files in examples dir
          //
          async.apply(fs.readdir, examplesDir),
          //
          // 2. If node_modules exists, continue. Otherwise
          //    exec `npm` to install them
          //
          function checkNodeModules(files, next) {
            if (files.indexOf('node_modules') !== -1) {
              return next();
            }

            var child = spawn('npm', ['install', '-f'], {
              cwd: examplesDir
            });

            child.on('exit', function (code) {
              return code
                ? next(new Error('npm install exited with non-zero exit code'))
                : next();
            });
          },
          //
          // 3. Read files in examples dir again to ensure the install
          //    worked as expected.
          //
          async.apply(fs.readdir, examplesDir),
        ], this.callback);
      },
      "examples/node_modules should exist": function (err, files) {
        assert.notEqual(files.indexOf('node_modules'), -1);
      }
    }
  }
};

//
// ### function shouldRequire (file)
// #### @file {string} File to attempt to require
//
// Returns a test which attempts to require `file`.
//
exports.shouldRequire = function (file) {
  return {
    "should have no errors": function () {
      try { assert.isObject(require(file)) }
      catch (ex) { assert.isNull(ex) }
    }
  };
};

//
// ### function shouldHaveNoErrors ()
//
// Returns a vows context that attempts to require
// every relevant example file in `examples`.
//
exports.shouldHaveNoErrors = function () {
  var context = {};

  ['balancer', 'http', 'middleware', 'websocket'].forEach(function (dir) {
    var name = 'examples/' + dir,
        files = fs.readdirSync(path.join(rootDir, 'examples', dir));

    files.forEach(function (file) {
      context[name + '/' + file] = exports.shouldRequire(path.join(
        examplesDir, dir, file
      ));
    });
  });

  return context;
};