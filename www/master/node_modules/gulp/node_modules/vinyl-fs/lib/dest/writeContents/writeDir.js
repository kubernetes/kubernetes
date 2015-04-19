'use strict';

var mkdirp = require('mkdirp');

function writeDir (writePath, file, cb) {
  mkdirp(writePath, file.stat.mode, cb);
}

module.exports = writeDir;
