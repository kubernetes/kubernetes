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

var sent_body = '';
var server_req_complete = false;
var client_res_complete = false;

var server = http.createServer(function (req, res) {
  assert.equal('POST', req.method);
  req.setEncoding('utf8');

  req.on('data', function (chunk) {
    console.log('server got: ' + JSON.stringify(chunk));
    sent_body += chunk;
  });

  req.on('end', function () {
    server_req_complete = true;
    console.log('request complete from server');
    res.writeHead(200, {'Content-Type': 'text/plain'});
    res.write('hello\n');
    res.end();
  });
});
server.listen(common.PORT);

server.on('listening', function () {
  var req = http.request({
    port: common.PROXY_PORT,
    method: 'POST',
    path: '/'
  }, function (res) {
    res.setEncoding('utf8');
    res.on('data', function (chunk) {
      console.log(chunk);
    });
    res.on('end', function () {
      client_res_complete = true;
      server.close();
    });
  });

  req.write('1\n');
  req.write('2\n');
  req.write('3\n');
  req.end();

  common.error('client finished sending request');
});

process.on('exit', function () {
  assert.equal('1\n2\n3\n', sent_body);
  assert.equal(true, server_req_complete);
  assert.equal(true, client_res_complete);
});
