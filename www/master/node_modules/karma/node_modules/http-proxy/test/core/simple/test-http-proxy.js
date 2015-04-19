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
var url = require('url');

var PROXY_PORT = common.PORT;
var BACKEND_PORT = common.PORT + 1;

var cookies = [
  'session_token=; path=/; expires=Sun, 15-Sep-2030 13:48:52 GMT',
  'prefers_open_id=; path=/; expires=Thu, 01-Jan-1970 00:00:00 GMT'
];

var headers = {'content-type': 'text/plain',
                'set-cookie': cookies,
                'hello': 'world' };

var backend = http.createServer(function (req, res) {
  common.debug('backend request');
  res.writeHead(200, headers);
  res.write('hello world\n');
  res.end();
});

var proxy = http.createServer(function (req, res) {
  common.debug('proxy req headers: ' + JSON.stringify(req.headers));
  var proxy_req = http.get({
    port: BACKEND_PORT,
    path: url.parse(req.url).pathname
  }, function (proxy_res) {

    common.debug('proxy res headers: ' + JSON.stringify(proxy_res.headers));

    assert.equal('world', proxy_res.headers['hello']);
    assert.equal('text/plain', proxy_res.headers['content-type']);
    assert.deepEqual(cookies, proxy_res.headers['set-cookie']);

    res.writeHead(proxy_res.statusCode, proxy_res.headers);

    proxy_res.on('data', function (chunk) {
      res.write(chunk);
    });

    proxy_res.on('end', function () {
      res.end();
      common.debug('proxy res');
    });
  });
});

var body = '';

var nlistening = 0;
function startReq() {
  nlistening++;
  if (nlistening < 2) return;

  var client = http.get({
    port: common.PROXY_PORT,
    path: '/test'
  }, function (res) {
    common.debug('got res');
    assert.equal(200, res.statusCode);

    assert.equal('world', res.headers['hello']);
    assert.equal('text/plain', res.headers['content-type']);
    assert.deepEqual(cookies, res.headers['set-cookie']);

    res.setEncoding('utf8');
    res.on('data', function (chunk) { body += chunk; });
    res.on('end', function () {
      proxy.close();
      backend.close();
      common.debug('closed both');
    });
  });
  common.debug('client req');
}

common.debug('listen proxy');
proxy.listen(PROXY_PORT, startReq);

common.debug('listen backend');
backend.listen(BACKEND_PORT, startReq);

process.on('exit', function () {
  assert.equal(body, 'hello world\n');
});
