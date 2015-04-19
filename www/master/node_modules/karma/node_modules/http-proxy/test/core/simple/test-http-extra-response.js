// Copyright Joyent, Inc. and other Node contributors.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit
// persons to whom the Software is furnished to do so, subject to the
// following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
// NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
// USE OR OTHER DEALINGS IN THE SOFTWARE.

var common = require('../common');
var assert = require('assert');
var http = require('http');
var net = require('net');

// If an HTTP server is broken and sends data after the end of the response,
// node should ignore it and drop the connection.
// Demos this bug: https://github.com/ry/node/issues/680

var body = 'hello world\r\n';
var fullResponse =
    'HTTP/1.1 500 Internal Server Error\r\n' +
    'Content-Length: ' + body.length + '\r\n' +
    'Content-Type: text/plain\r\n' +
    'Date: Fri + 18 Feb 2011 06:22:45 GMT\r\n' +
    'Host: 10.20.149.2\r\n' +
    'Access-Control-Allow-Credentials: true\r\n' +
    'Server: badly broken/0.1 (OS NAME)\r\n' +
    '\r\n' +
    body;

var gotResponse = false;


var server = net.createServer(function (socket) {
  var postBody = '';

  socket.setEncoding('utf8');

  socket.on('data', function (chunk) {
    postBody += chunk;

    if (postBody.indexOf('\r\n') > -1) {
      socket.write(fullResponse);
      // omg, I wrote the response twice, what a terrible HTTP server I am.
      socket.end(fullResponse);
    }
  });
});


server.listen(common.PORT, function () {
  http.get({ port: common.PROXY_PORT }, function (res) {
    var buffer = '';
    console.log('Got res code: ' + res.statusCode);

    res.setEncoding('utf8');
    res.on('data', function (chunk) {
      buffer += chunk;
    });

    res.on('end', function () {
      console.log('Response ended, read ' + buffer.length + ' bytes');
      assert.equal(body, buffer);
      server.close();
      gotResponse = true;
    });
  });
});


process.on('exit', function () {
  assert.ok(gotResponse);
});

