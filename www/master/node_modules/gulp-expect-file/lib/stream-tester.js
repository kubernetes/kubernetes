'use strict';

var through = require('through2');
var StringTester = require('./string-tester');

module.exports = StreamTester;

function StreamTester(contentsTester) {
  if (contentsTester instanceof StringTester) {
    this.contentsTester = contentsTester;
  } else {
    this.contentsTester = new StringTester(contentsTester);
  }
}

StreamTester.prototype.test = function (stream, callback) {
  var spyStream = this.createStream();
  var yielded = false;
  return stream.pipe(spyStream)
    .on('error', function (err) {
      yielded || callback(err);
      yielded = true;
    })
    .on('end', function () {
      yielded || callback(null);
      yielded = true;
    });
};

StreamTester.prototype.createStream = function () {
  var contentsTester = this.contentsTester;
  var contents = '';
  return through(
    function (chunk, encoding, cb) {
      contents += chunk.toString();
      this.push(chunk);
      return cb();
    },
    function (cb) {
      var _this = this;
      contentsTester.test(contents, function (err) {
        _this.emit('error', err);
        cb();
      });
    }
  );
};
