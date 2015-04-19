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

// This tests setTimeout() by having multiple clients connecting and sending
// data in random intervals. Clients are also randomly disconnecting until there
// are no more clients left. If no false timeout occurs, this test has passed.
var common = require('../common'),
    assert = require('assert'),
    http = require('http'),
    server = http.createServer(),
    connections = 0;

server.on('request', function (req, res) {
  req.socket.setTimeout(1000);
  req.socket.on('timeout', function () {
    throw new Error('Unexpected timeout');
  });
  req.on('end', function () {
    connections--;
    res.writeHead(200);
    res.end('done\n');
    if (connections == 0) {
      server.close();
    }
  });
});

server.listen(common.PORT, '127.0.0.1', function () {
  for (var i = 0; i < 10; i++) {
    connections++;

    setTimeout(function () {
      var request = http.request({
        port: common.PROXY_PORT,
        method: 'POST',
        path: '/'
      });

      function ping() {
        var nextPing = (Math.random() * 900).toFixed();
        if (nextPing > 600) {
          request.end();
          return;
        }
        request.write('ping');
        setTimeout(ping, nextPing);
      }
      ping();
    }, i * 50);
  }
});
