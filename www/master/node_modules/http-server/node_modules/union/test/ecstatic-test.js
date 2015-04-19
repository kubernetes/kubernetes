/*
 * simple-test.js: Simple tests for basic streaming and non-streaming HTTP requests with union.
 *
 * (C) 2011, Nodejitsu Inc.
 * MIT LICENSE
 *
 */

var assert = require('assert'),
    ecstatic = require('ecstatic')(__dirname + '/fixtures/static'),
    request = require('request'),
    vows = require('vows'),
    union = require('../');

vows.describe('union/ecstatic').addBatch({
  "When using union with ecstatic": {
    topic: function () {
      union.createServer({
        before: [
          ecstatic
        ]
      }).listen(18082, this.callback);
    },
    "a request to /some-file.txt": {
      topic: function () {
        request({ uri: 'http://localhost:18082/some-file.txt' }, this.callback);
      },
      "should respond with `hello world`": function (err, res, body) {
        assert.isNull(err);
        assert.equal(body, 'hello world\n');
      }
    },
    "a request to /404.txt (which does not exist)": {
      topic: function () {
        request({ uri: 'http://localhost:18082/404.txt' }, this.callback);
      },
      "should respond with 404 status code": function (err, res, body) {
        assert.isNull(err);
        assert.equal(res.statusCode, 404);
      }
    }
  }
}).export(module);

