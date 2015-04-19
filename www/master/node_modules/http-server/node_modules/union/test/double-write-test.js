/*
 * simple-test.js: Simple tests for basic streaming and non-streaming HTTP requests with union.
 *
 * (C) 2011, Nodejitsu Inc.
 * MIT LICENSE
 *
 */

var assert = require('assert'),
    fs = require('fs'),
    path = require('path'),
    request = require('request'),
    vows = require('vows'),
    union = require('../lib/index'),
    macros = require('./helpers/macros');

var doubleWrite = false,
    server;
    
server = union.createServer({
  before: [
    function (req, res) {
      res.json(200, { 'hello': 'world' });
      res.emit('next');
    },
    function (req, res) {
      doubleWrite = true;
      res.json(200, { 'hello': 'world' });
      res.emit('next');
    }    
  ]
});


vows.describe('union/double-write').addBatch({
  "When using union": {
    "an http server which attempts to write to the response twice": {
      topic: function () {
        server.listen(9091, this.callback);
      },
      "a GET request to `/foo`": {
        topic: function () {
          request({ uri: 'http://localhost:9091/foo' }, this.callback);
        },
        "it should respond with `{ 'hello': 'world' }`": function (err, res, body) {
          macros.assertValidResponse(err, res);
          assert.deepEqual(JSON.parse(body), { 'hello': 'world' });
        },
        "it should not write to the response twice": function () {
          assert.isFalse(doubleWrite);
        }
      }
    }
  }
}).addBatch({
  "When the tests are over": {
    "the server should close": function () {
      server.close();
    }
  }
}).export(module);

