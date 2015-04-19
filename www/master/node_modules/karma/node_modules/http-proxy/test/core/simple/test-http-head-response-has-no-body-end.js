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

// libuv-broken


var common = require('../common');
var assert = require('assert');

var http = require('http');

// This test is to make sure that when the HTTP server
// responds to a HEAD request with data to res.end,
// it does not send any body.

var server = http.createServer(function (req, res) {
  res.writeHead(200);
  res.end('FAIL'); // broken: sends FAIL from hot path.
});
server.listen(common.PORT);

var responseComplete = false;

server.on('listening', function () {
  var req = http.request({
    port: common.PROXY_PORT,
    method: 'HEAD',
    path: '/'
  }, function (res) {
    common.error('response');
    res.on('end', function () {
      common.error('response end');
      server.close();
      responseComplete = true;
    });
  });
  common.error('req');
  req.end();
});

process.on('exit', function () {
  assert.ok(responseComplete);
});
