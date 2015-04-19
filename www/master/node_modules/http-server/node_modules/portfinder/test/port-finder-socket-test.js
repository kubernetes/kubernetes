/*
 * portfinder-test.js: Tests for the `portfinder` module.
 *
 * (C) 2011, Charlie Robbins
 *
 */

var assert = require('assert'),
    exec = require('child_process').exec,
    net = require('net'),
    path = require('path'),
    async = require('async'),
    vows = require('vows'),
    portfinder = require('../lib/portfinder');

var servers = [],
    socketDir = path.join(__dirname, 'fixtures'),
    badDir = path.join(__dirname, 'bad-dir');

function createServers (callback) {
  var base = 0;
  
  async.whilst(
    function () { return base < 5 },
    function (next) {
      var server = net.createServer(function () { }),
          name = base === 0 ? 'test.sock' : 'test' + base + '.sock';
      
      server.listen(path.join(socketDir, name), next);
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
          portfinder.getSocket({
            path: path.join(socketDir, 'test.sock')
          }, this.callback);
        },
        "should respond with the first free socket (test5.sock)": function (err, socket) {
          assert.isTrue(!err);
          assert.equal(socket, path.join(socketDir, 'test5.sock'));
        }
      }
    }
  }
}).addBatch({
  "When using portfinder module": {
    "with no existing servers": {
      "the getSocket() method": {
        "with a directory that doesnt exist": {
          topic: function () {
            var that = this;
            exec('rm -rf ' + badDir, function () {
              portfinder.getSocket({
                path: path.join(badDir, 'test.sock')
              }, that.callback);
            });
          },
          "should respond with the first free socket (test.sock)": function (err, socket) {
            assert.isTrue(!err);
            assert.equal(socket, path.join(badDir, 'test.sock'));
          }
        },
        "with a directory that exists": {
          topic: function () {
            portfinder.getSocket({
              path: path.join(socketDir, 'exists.sock')
            }, this.callback);
          },
          "should respond with the first free socket (exists.sock)": function (err, socket) {
            assert.isTrue(!err);
            assert.equal(socket, path.join(socketDir, 'exists.sock'));
          }
        }
      }
    }
  }
}).addBatch({
  "When the tests are over": {
    "necessary cleanup should take place": function () {
      exec('rm -rf ' + badDir + ' ' + path.join(socketDir, '*'), function () { });
    }
  }
}).export(module);