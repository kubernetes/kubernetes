var
  fs = require('fs'),
  tmp = require('../lib/tmp');

function _writeSync(stream, str, cb) {
  var flushed = stream.write(str);
  if (flushed) {
    return cb(null);
  }

  stream.once('drain', function _flushed() {
    cb(null);
  });
}

module.exports.out = function (str, cb) {
  _writeSync(process.stdout, str, cb);
};

module.exports.err = function (str, cb) {
  _writeSync(process.stderr, str, cb);
};

module.exports.exit = function () {
  process.exit(0);
};

var type = process.argv[2];
module.exports.tmpFunction = (type == 'file') ? tmp.file : tmp.dir;

var arg = (process.argv[3] && parseInt(process.argv[3], 10) === 1) ? true : false;
module.exports.arg = arg;
