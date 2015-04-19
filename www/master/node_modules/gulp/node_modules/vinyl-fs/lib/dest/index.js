'use strict';

var defaults = require('defaults');
var path = require('path');
var through2 = require('through2');
var mkdirp = require('mkdirp');
var fs = require('graceful-fs');

var writeContents = require('./writeContents');


function dest(outFolder, opt) {
  opt = opt || {};
  if (typeof outFolder !== 'string' && typeof outFolder !== 'function') {
    throw new Error('Invalid output folder');
  }

  var options = defaults(opt, {
    cwd: process.cwd()
  });

  if (typeof options.mode === 'string') {
    options.mode = parseInt(options.mode, 8);
  }

  var cwd = path.resolve(options.cwd);

  function saveFile (file, enc, cb) {
    var basePath;
    if (typeof outFolder === 'string') {
      basePath = path.resolve(cwd, outFolder);
    }
    if (typeof outFolder === 'function') {
      basePath = path.resolve(cwd, outFolder(file));
    }
    var writePath = path.resolve(basePath, file.relative);
    var writeFolder = path.dirname(writePath);

    // wire up new properties
    file.stat = file.stat ? file.stat : new fs.Stats();
    file.stat.mode = (options.mode || file.stat.mode);
    file.cwd = cwd;
    file.base = basePath;
    file.path = writePath;

    // mkdirp the folder the file is going in
    mkdirp(writeFolder, function(err){
      if (err) {
        return cb(err);
      }
      writeContents(writePath, file, cb);
    });
  }

  var stream = through2.obj(saveFile);
  // TODO: option for either backpressure or lossy
  stream.resume();
  return stream;
}

module.exports = dest;
