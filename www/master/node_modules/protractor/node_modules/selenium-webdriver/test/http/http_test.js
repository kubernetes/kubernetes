// Copyright 2014 Selenium committers
// Copyright 2014 Software Freedom Conservancy
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//     You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

var assert = require('assert');
var http = require('http');

var HttpClient = require('../../http').HttpClient;
var HttpRequest = require('../../_base').require('webdriver.http.Request');
var Server = require('../../lib/test/httpserver').Server;
var promise = require('../..').promise;
var test = require('../../lib/test');

describe('HttpClient', function() {
  var server = new Server(function(req, res) {
    if (req.method == 'GET' && req.url == '/echo') {
      res.writeHead(200, req.headers);
      res.end();

    } else if (req.method == 'GET' && req.url == '/redirect') {
      res.writeHead(303, {'Location': server.url('/hello')});
      res.end();

    } else if (req.method == 'GET' && req.url == '/hello') {
      res.writeHead(200, {'content-type': 'text/plain'});
      res.end('hello, world!');

    } else if (req.method == 'GET' && req.url == '/badredirect') {
      res.writeHead(303, {});
      res.end();

    } else {
      res.writeHead(404, {});
      res.end();
    }
  });

  test.before(server.start.bind(server));
  test.after(server.stop.bind(server));

  test.it('can send a basic HTTP request', function() {
    var request = new HttpRequest('GET', '/echo');
    request.headers['Foo'] = 'Bar';

    var agent = new http.Agent();
    agent.maxSockets = 1;  // Only making 1 request.

    var client = new HttpClient(server.url(), agent);
    return promise.checkedNodeCall(client.send.bind(client, request))
        .then(function(response) {
          assert.equal(200, response.status);
          assert.equal(
            'application/json; charset=utf-8', response.headers['accept']);
          assert.equal('Bar', response.headers['foo']);
          assert.equal('0', response.headers['content-length']);
          assert.equal('keep-alive', response.headers['connection']);
          assert.equal(server.host(), response.headers['host']);
        });
  });

  test.it('automatically follows redirects', function() {
    var request = new HttpRequest('GET', '/redirect');
    var client = new HttpClient(server.url());
    return promise.checkedNodeCall(client.send.bind(client, request))
        .then(function(response) {
          assert.equal(200, response.status);
          assert.equal('text/plain', response.headers['content-type']);
          assert.equal('hello, world!', response.body);
        });
  });

  test.it('handles malformed redirect responses', function() {
    var request = new HttpRequest('GET', '/badredirect');
    var client = new HttpClient(server.url());
    return promise.checkedNodeCall(client.send.bind(client, request)).
        thenCatch(function(err) {
          assert.ok(/Failed to parse "Location"/.test(err.message),
              'Not the expected error: ' + err.message);
        });
  });
});
