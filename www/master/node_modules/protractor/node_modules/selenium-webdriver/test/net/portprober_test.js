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
    net = require('net');

var promise = require('../..').promise,
    portprober = require('../../net/portprober');

describe('isFree', function() {

  var server;

  beforeEach(function() {
    server = net.createServer(function(){});
  })

  afterEach(function(done) {
    if (!server) return done();
    server.close(function() {
      done();
    });
  });

  it('should work for INADDR_ANY', function(done) {
    server.listen(0, function() {
      var port = server.address().port;
      assertPortNotfree(port).then(function() {
        var done = promise.defer();
        server.close(function() {
          server = null;
          done.fulfill(assertPortIsFree(port));
        });
        return done.promise;
      }).then(function() { done(); }, done);
    });
  });

  it('should work for a specific host', function(done) {
    var host = '127.0.0.1';
    server.listen(0, host, function() {
      var port = server.address().port;
      assertPortNotfree(port, host).then(function() {
        var done = promise.defer();
        server.close(function() {
          server = null;
          done.fulfill(assertPortIsFree(port, host));
        });
        return done.promise;
      }).then(function() { done(); }, done);
    });
  });
});

describe('findFreePort', function() {
  var server;

  beforeEach(function() {
    server = net.createServer(function(){});
  })

  afterEach(function(done) {
    if (!server) return done();
    server.close(function() {
      done();
    });
  });

  it('should work for INADDR_ANY', function(done) {
    portprober.findFreePort().then(function(port) {
      server.listen(port, function() {
        assertPortNotfree(port).then(function() {
          var done = promise.defer();
          server.close(function() {
            server = null;
            done.fulfill(assertPortIsFree(port));
          });
          return done.promise;
        }).then(function() { done(); }, done);
      });
    });
  });

  it('should work for a specific host', function(done) {
    var host = '127.0.0.1';
    portprober.findFreePort(host).then(function(port) {
      server.listen(port, host, function() {
        assertPortNotfree(port, host).then(function() {
          var done = promise.defer();
          server.close(function() {
            server = null;
            done.fulfill(assertPortIsFree(port, host));
          });
          return done.promise;
        }).then(function() { done(); }, done);
      });
    });
  });
});


function assertPortIsFree(port, opt_host) {
  return portprober.isFree(port, opt_host).then(function(free) {
    assert.ok(free, 'port should be free');
  });
}


function assertPortNotfree(port, opt_host) {
  return portprober.isFree(port, opt_host).then(function(free) {
    assert.ok(!free, 'port should is not free');
  });
}
