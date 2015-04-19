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

function p(x) {
  common.error(common.inspect(x));
}

var responses_sent = 0;
var responses_recvd = 0;
var body0 = '';
var body1 = '';

var server = http.Server(function (req, res) {
  if (responses_sent == 0) {
    assert.equal('GET', req.method);
    assert.equal('/hello', url.parse(req.url).pathname);

    console.dir(req.headers);
    assert.equal(true, 'accept' in req.headers);
    assert.equal('*/*', req.headers['accept']);

    assert.equal(true, 'foo' in req.headers);
    assert.equal('bar', req.headers['foo']);
  }

  if (responses_sent == 1) {
    assert.equal('POST', req.method);
    assert.equal('/world', url.parse(req.url).pathname);
    this.close();
  }

  req.on('end', function () {
    res.writeHead(200, {'Content-Type': 'text/plain'});
    res.write('The path was ' + url.parse(req.url).pathname);
    res.end();
    responses_sent += 1;
  });

  //assert.equal('127.0.0.1', res.connection.remoteAddress);
});
server.listen(common.PORT);

server.on('listening', function () {
  var agent = new http.Agent({ port: common.PROXY_PORT, maxSockets: 1 });
  http.get({
    port: common.PROXY_PORT,
    path: '/hello',
    headers: {'Accept': '*/*', 'Foo': 'bar'},
    agent: agent
  }, function (res) {
    assert.equal(200, res.statusCode);
    responses_recvd += 1;
    res.setEncoding('utf8');
    res.on('data', function (chunk) { body0 += chunk; });
    common.debug('Got /hello response');
  });

  setTimeout(function () {
    var req = http.request({
      port: common.PROXY_PORT,
      method: 'POST',
      path: '/world',
      agent: agent
    }, function (res) {
      assert.equal(200, res.statusCode);
      responses_recvd += 1;
      res.setEncoding('utf8');
      res.on('data', function (chunk) { body1 += chunk; });
      common.debug('Got /world response');
    });
    req.end();
  }, 1);
});

process.on('exit', function () {
  common.debug('responses_recvd: ' + responses_recvd);
  assert.equal(2, responses_recvd);

  common.debug('responses_sent: ' + responses_sent);
  assert.equal(2, responses_sent);

  assert.equal('The path was /hello', body0);
  assert.equal('The path was /world', body1);
});

