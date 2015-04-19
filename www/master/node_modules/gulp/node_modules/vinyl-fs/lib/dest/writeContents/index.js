'use strict';

var fs = require('fs');
var writeDir = require('./writeDir');
var writeStream = require('./writeStream');
var writeBuffer = require('./writeBuffer');

function writeContents(writePath, file, cb) {
  var written = function(err) {
    var done = function(err) {
      cb(err, file);
    };
    if (err) {
      return done(err);
    }

    if (!file.stat || typeof file.stat.mode !== 'number') {
      return done();
    }

    fs.stat(writePath, function(err, st) {
      if (err) {
        return done(err);
      }
      // octal 7777 = decimal 4095
      var currentMode = (st.mode & 4095);
      if (currentMode === file.stat.mode) {
        return done();
      }
      fs.chmod(writePath, file.stat.mode, done);
    });
  };

  // if directory then mkdirp it
  if (file.isDirectory()) {
    writeDir(writePath, file, written);
    return;
  }

  // stream it to disk yo
  if (file.isStream()) {
    writeStream(writePath, file, written);
    return;
  }

  // write it like normal
  if (file.isBuffer()) {
    writeBuffer(writePath, file, written);
    return;
  }

  // if no contents then do nothing
  if (file.isNull()) {
    cb(null, file);
    return;
  }
}

module.exports = writeContents;
