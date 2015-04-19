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

var clientAborts = 0;

var server = http.Server(function (req, res) {
  console.log('Got connection');
  res.writeHead(200);
  res.write('Working on it...');

  // I would expect an error event from req or res that the client aborted
  // before completing the HTTP request / response cycle, or maybe a new
  // event like "aborted" or something.
  req.on('aborted', function () {
    clientAborts++;
    console.log('Got abort ' + clientAborts);
    if (clientAborts === N) {
      console.log('All aborts detected, you win.');
      server.close();
    }
  });

  // since there is already clientError, maybe that would be appropriate,
  // since "error" is magical
  req.on('clientError', function () {
    console.log('Got clientError');
  });
});

var responses = 0;
var N = http.Agent.defaultMaxSockets - 1;
var requests = [];

server.listen(common.PORT, function () {
  console.log('Server listening.');

  for (var i = 0; i < N; i++) {
    console.log('Making client ' + i);
    var options = { port: common.PROXY_PORT, path: '/?id=' + i };
    var req = http.get(options, function (res) {
      console.log('Client response code ' + res.statusCode);

      if (++responses == N) {
        console.log('All clients connected, destroying.');
        requests.forEach(function (outReq) {
          console.log('abort');
          outReq.abort();
        });
      }
    });

    requests.push(req);
  }
});

process.on('exit', function () {
  assert.equal(N, clientAborts);
});
