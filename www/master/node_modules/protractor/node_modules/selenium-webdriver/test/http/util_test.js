// Copyright 2013 Selenium committers
// Copyright 2013 Software Freedom Conservancy
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

'use strict';

var assert = require('assert'),
    http = require('http');

var util = require('../../http/util');

describe('selenium-webdriver/http/util', function() {

  var server, baseUrl;

  var status, value, responseCode;

  function startServer(done) {
    if (server) return done();

    server = http.createServer(function(req, res) {
      var data = JSON.stringify({status: status, value: value});
      res.writeHead(responseCode, {
        'Content-Type': 'application/json; charset=utf-8',
        'Content-Length': Buffer.byteLength(data, 'utf8')
      });
      res.end(data);
    });

    server.listen(0, '127.0.0.1', function(e) {
      if (e) return done(e);

      var addr = server.address();
      baseUrl = 'http://' + addr.address + ':' + addr.port;
      done();
    });
  }

  function killServer(done) {
    if (!server) return done();
    server.close(done);
    server = null;
  }

  after(killServer);

  beforeEach(function(done) {
    status = 0;
    value = 'abc123';
    responseCode = 200;
    startServer(done);
  });

  describe('#getStatus', function() {
    it('should return value field on success', function(done) {
      util.getStatus(baseUrl).then(function(response) {
        assert.equal('abc123', response);
      }).thenFinally(done);
    });

    it('should fail if response object is not success', function(done) {
      status = 1;
      util.getStatus(baseUrl).then(function() {
        throw Error('expected a failure');
      }, function(err) {
        assert.equal(status, err.code);
        assert.equal(value, err.message);
      }).thenFinally(done);
    });

    it('should fail if the server is not listening', function(done) {
      killServer(function(e) {
        if(e) return done(e);

        util.getStatus(baseUrl).then(function() {
          throw Error('expected a failure');
        }, function() {
          // Expected.
        }).thenFinally(done);
      });
    });

    it('should fail if HTTP status is not 200', function(done) {
      status = 1;
      responseCode = 404;
      util.getStatus(baseUrl).then(function() {
        throw Error('expected a failure');
      }, function(err) {
        assert.equal(status, err.code);
        assert.equal(value, err.message);
      }).thenFinally(done);
    });
  });

  describe('#waitForServer', function() {
    it('resolves when server is ready', function(done) {
      status = 1;
      setTimeout(function() { status = 0; }, 50);
      util.waitForServer(baseUrl, 100).
          then(function() {}).  // done needs no argument to pass.
          thenFinally(done);
    });

    it('should fail if server does not become ready', function(done) {
      status = 1;
      util.waitForServer(baseUrl, 50).
          then(function() { done('Expected to time out'); },
               function() { done(); });
    });

    it('can cancel wait', function(done) {
      status = 1;
      var err = Error('cancelled!');
      var isReady =  util.waitForServer(baseUrl, 200).
          then(function() { done('Did not expect to succeed'); }).
          then(null, function(e) {
            assert.equal('cancelled!', e.message);
          }).
          then(function() { done(); }, done);

      setTimeout(function() {
        isReady.cancel('cancelled!');
      }, 50);
    });
  });

  describe('#waitForUrl', function() {
    it('succeeds when URL returns 2xx', function(done) {
      responseCode = 404;
      setTimeout(function() { responseCode = 200; }, 50);

      util.waitForUrl(baseUrl, 200).
          then(function() {}).  // done needs no argument to pass.
          thenFinally(done);
    });

    it('fails if URL always returns 4xx', function(done) {
      responseCode = 404;

      util.waitForUrl(baseUrl, 50).
          then(function() { done('Expected to time out'); },
               function() { done(); });
    });

    it('fails if cannot connect to server', function(done) {
      killServer(function(e) {
        if (e) return done(e);

      util.waitForUrl(baseUrl, 50).
          then(function() { done('Expected to time out'); },
               function() { done(); });
      });
    });

    it('can cancel wait', function(done) {
      responseCode = 404;
      var isReady =  util.waitForUrl(baseUrl, 200).
          then(function() { done('Did not expect to succeed'); }).
          then(null, function(e) {
            assert.equal('cancelled!', e.message);
          }).
          then(function() { done(); }, done);

      setTimeout(function() {
        isReady.cancel('cancelled!');
      }, 50);
    });
  });
});
