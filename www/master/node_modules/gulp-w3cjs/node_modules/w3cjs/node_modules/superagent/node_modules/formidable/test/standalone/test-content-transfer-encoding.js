var assert = require('assert');
var common = require('../common');
var formidable = require('../../lib/index');
var http = require('http');

var server = http.createServer(function(req, res) {
  var form = new formidable.IncomingForm();
  form.uploadDir = common.dir.tmp;
  form.on('end', function () {
    throw new Error('Unexpected "end" event');
  });
  form.on('error', function (e) {
    res.writeHead(500);
    res.end(e.message);
  });
  form.parse(req);
});

server.listen(0, function() {
  var body =
    '--foo\r\n' +
    'Content-Disposition: form-data; name="file1"; filename="file1"\r\n' +
    'Content-Type: application/octet-stream\r\n' +
    '\r\nThis is the first file\r\n' +
    '--foo\r\n' +
    'Content-Type: application/octet-stream\r\n' +
    'Content-Disposition: form-data; name="file2"; filename="file2"\r\n' +
    'Content-Transfer-Encoding: unknown\r\n' +
    '\r\nThis is the second file\r\n' +
    '--foo--\r\n';

  var req = http.request({
    method: 'POST',
    port: server.address().port,
    headers: {
      'Content-Length': body.length,
      'Content-Type': 'multipart/form-data; boundary=foo'
    }
  });
  req.on('response', function (res) {
    assert.equal(res.statusCode, 500);
    res.on('data', function () {});
    res.on('end', function () {
      server.close();
    });
  });
  req.end(body);
});
