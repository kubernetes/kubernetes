/*
 * portfinder-test.js: Tests for the `portfinder` module.
 *
 * (C) 2011, Charlie Robbins
 *
 */

var vows = require('vows'),
    assert = require('assert'),
    async = require('async'),
    http = require('http'),
    portfinder = require('../lib/portfinder');

var servers = [];

function createServers (callback) {
  var base = 8000;
  
  async.whilst(
    function () { return base < 8005 },
    function (next) {
      var server = http.createServer(function () { });
      server.listen(base, next);
      base++;
      servers.push(server);
    }, callback);
}

vows.describe('portfinder').addBatch({
  "When using portfinder module": {
    "with 5 existing servers": {
      topic: function () {
        createServers(this.callback);
      },
      "the getPort() method": {
        topic: function () {
          portfinder.getPort(this.callback);
        },
        "should respond with the first free port (8005)": function (err, port) {
          assert.isTrue(!err);
          assert.equal(port, 8005);
        }
      }
    }
  }
}).addBatch({
  "When using portfinder module": {
    "with no existing servers": {
      topic: function () {
        servers.forEach(function (server) {
          server.close();
        });
        
        return null;
      },
      "the getPort() method": {
        topic: function () {
          portfinder.getPort(this.callback);
        },
        "should respond with the first free port (8000)": function (err, port) {
          assert.isTrue(!err);
          assert.equal(port, 8000);
        }
      }
    }
  }
}).export(module);