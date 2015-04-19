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


var http = require('http'),
    common = require('../common'),
    assert = require('assert'),
    httpServer = http.createServer(reqHandler);

function reqHandler(req, res) {
  console.log('Got request: ' + req.headers.host + ' ' + req.url);
  if (req.url === '/setHostFalse5') {
    assert.equal(req.headers.host, undefined);
  } else {
    assert.equal(req.headers.host, 'localhost:' + common.PROXY_PORT,
                 'Wrong host header for req[' + req.url + ']: ' +
                 req.headers.host);
  }
  res.writeHead(200, {});
  //process.nextTick(function () { res.end('ok'); });
  res.end('ok');
}

function thrower(er) {
  throw er;
}

testHttp();

function testHttp() {

  console.log('testing http on port ' + common.PROXY_PORT + ' (proxied to ' +
              common.PORT + ')');

  var counter = 0;

  function cb() {
    counter--;
    console.log('back from http request. counter = ' + counter);
    if (counter === 0) {
      httpServer.close();
    }
  }

  httpServer.listen(common.PORT, function (er) {
    console.error('listening on ' + common.PORT);

    if (er) throw er;

    http.get({ method: 'GET',
      path: '/' + (counter++),
      host: 'localhost',
      //agent: false,
      port: common.PROXY_PORT }, cb).on('error', thrower);

    http.request({ method: 'GET',
      path: '/' + (counter++),
      host: 'localhost',
      //agent: false,
      port: common.PROXY_PORT }, cb).on('error', thrower).end();

    http.request({ method: 'POST',
      path: '/' + (counter++),
      host: 'localhost',
      //agent: false,
      port: common.PROXY_PORT }, cb).on('error', thrower).end();

    http.request({ method: 'PUT',
      path: '/' + (counter++),
      host: 'localhost',
      //agent: false,
      port: common.PROXY_PORT }, cb).on('error', thrower).end();

    http.request({ method: 'DELETE',
      path: '/' + (counter++),
      host: 'localhost',
      //agent: false,
      port: common.PROXY_PORT }, cb).on('error', thrower).end();
  });
}

