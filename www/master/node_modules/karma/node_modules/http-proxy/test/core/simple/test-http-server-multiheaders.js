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

// Verify that the HTTP server implementation handles multiple instances
// of the same header as per RFC2616: joining the handful of fields by ', '
// that support it, and dropping duplicates for other fields.

var common = require('../common');
var assert = require('assert');
var http = require('http');

var srv = http.createServer(function (req, res) {
  assert.equal(req.headers.accept, 'abc, def, ghijklmnopqrst');
  assert.equal(req.headers.host, 'foo');
  assert.equal(req.headers['x-foo'], 'bingo');
  assert.equal(req.headers['x-bar'], 'banjo, bango');

  res.writeHead(200, {'Content-Type' : 'text/plain'});
  res.end('EOF');

  srv.close();
});

srv.listen(common.PORT, function () {
  http.get({
    host: 'localhost',
    port: common.PROXY_PORT,
    path: '/',
    headers: [
      ['accept', 'abc'],
      ['accept', 'def'],
      ['Accept', 'ghijklmnopqrst'],
      ['host', 'foo'],
      ['Host', 'bar'],
      ['hOst', 'baz'],
      ['x-foo', 'bingo'],
      ['x-bar', 'banjo'],
      ['x-bar', 'bango']
    ]
  });
});
