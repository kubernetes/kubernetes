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

var server = http.createServer(function (req, res) {
  common.error('got req');
  throw new Error('This shouldn\'t happen.');
});

server.on('upgrade', function (req, socket, upgradeHead) {
  common.error('got upgrade event');
  // test that throwing an error from upgrade gets
  // is uncaught
  throw new Error('upgrade error');
});

var gotError = false;

process.on('uncaughtException', function (e) {
  common.error('got \'clientError\' event');
  assert.equal('upgrade error', e.message);
  gotError = true;
  process.exit(0);
});


server.listen(common.PORT, function () {
  var c = net.createConnection(common.PROXY_PORT);

  c.on('connect', function () {
    common.error('client wrote message');
    c.write('GET /blah HTTP/1.1\r\n' +
            'Upgrade: WebSocket\r\n' +
            'Connection: Upgrade\r\n' +
            '\r\n\r\nhello world');
  });

  c.on('end', function () {
    c.end();
  });

  c.on('close', function () {
    common.error('client close');
    server.close();
  });
});

process.on('exit', function () {
  assert.ok(gotError);
});
