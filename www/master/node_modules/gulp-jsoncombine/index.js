"use strict";
var through = require('through');
var path = require('path');
var gutil = require('gulp-util');
var PluginError = gutil.PluginError;
var File = gutil.File;

module.exports = function (fileName, converter) {
  if (!fileName) {
    throw new PluginError('gulp-jsoncombine', 'Missing fileName option for gulp-jsoncombine');
  }
  if (!converter) {
    throw new PluginError('gulp-jsoncombine', 'Missing converter option for gulp-jsoncombine');
  }

  var data = {};
  var firstFile = null;

  function bufferContents(file) {
    if (!firstFile) {
      firstFile = file;
    }

    if (file.isNull()) {
      return; // ignore
    }
    if (file.isStream()) {
      return this.emit('error', new PluginError('gulp-jsoncombine', 'Streaming not supported'));
    }
    try {
      data[file.relative.substr(0,file.relative.length-5)] = JSON.parse(file.contents.toString());
    } catch (err) {
      return this.emit('error',
          new PluginError('gulp-jsoncombine', 'Error parsing JSON: ' + err));
    }
  }

  function endStream() {
    if (firstFile) {
      var joinedPath = path.join(firstFile.base, fileName);

      var joinedFile = new File({
        cwd: firstFile.cwd,
        base: firstFile.base,
        path: joinedPath,
        contents: converter(data)
      });

      this.emit('data', joinedFile);
    }
    this.emit('end');
  }

  return through(bufferContents, endStream);
};
