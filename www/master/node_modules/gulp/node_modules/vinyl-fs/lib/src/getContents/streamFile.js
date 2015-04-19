'use strict';

var fs = require('graceful-fs');
var stripBom = require('strip-bom');

function streamFile(file, cb) {
  file.contents = fs.createReadStream(file.path)
    .pipe(stripBom.stream());

  cb(null, file);
}

module.exports = streamFile;
