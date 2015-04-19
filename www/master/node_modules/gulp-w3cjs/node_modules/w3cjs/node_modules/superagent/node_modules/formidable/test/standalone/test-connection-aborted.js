var assert = require('assert');
var http = require('http');
var net = require('net');
var formidable = require('../../lib/index');

var server = http.createServer(function (req, res) {
  var form = new formidable.IncomingForm();
  var aborted_received = false;
  form.on('aborted', function () {
    aborted_received = true;
  });
  form.on('error', function () {
    assert(aborted_received, 'Error event should follow aborted');
    server.close();
  });
  form.on('end', function () {
    throw new Error('Unexpected "end" event');
  });
  form.parse(req);
}).listen(0, 'localhost', function () {
  var client = net.connect(server.address().port);
  client.write(
    "POST / HTTP/1.1\r\n" +
    "Content-Length: 70\r\n" +
    "Content-Type: multipart/form-data; boundary=foo\r\n\r\n");
  client.end();
});
