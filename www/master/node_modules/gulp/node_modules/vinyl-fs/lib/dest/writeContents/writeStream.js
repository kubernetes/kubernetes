'use strict';

var streamFile = require('../../src/getContents/streamFile');
var fs = require('graceful-fs');

function writeStream (writePath, file, cb) {
  var opt = {
    mode: file.stat.mode
  };

  var outStream = fs.createWriteStream(writePath, opt);

  file.contents.once('error', cb);
  outStream.once('error', cb);
  outStream.once('finish', function() {
    streamFile(file, cb);
  });

  file.contents.pipe(outStream);
}

module.exports = writeStream;
