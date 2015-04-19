'use strict';

var fs = require('graceful-fs');
var stripBom = require('strip-bom');

function bufferFile(file, cb) {
  fs.readFile(file.path, function (err, data) {
    if (err) {
      return cb(err);
    }
    file.contents = stripBom(data);
    cb(null, file);
  });
}

module.exports = bufferFile;
