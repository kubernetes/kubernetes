/*
 * simple-test.js: Simple tests for basic streaming and non-streaming HTTP requests with union.
 *
 * (C) 2011, Nodejitsu Inc.
 * MIT LICENSE
 *
 */

var assert = require('assert'),
    bodyParser = require('connect').bodyParser,
    request = require('request'),
    vows = require('vows'),
    union = require('../');

vows.describe('union/body-parser').addBatch({
  "When using union with connect bodyParser()": {
    topic: function () {
      union.createServer({
        buffer: false,
        before: [
          bodyParser(),
          function (req, res) {
            res.end(JSON.stringify(req.body, true, 2));
          }
        ]
      }).listen(8082, this.callback);
    },
    "a request to /": {
      topic: function () {
        request.post({
          uri: 'http://localhost:8082/',
          headers: {
            'content-type': 'application/json'
          },
          body: JSON.stringify({ a: "foo", b: "bar" })
        }, this.callback);
      },
      "should respond with a body-decoded object": function (err, res, body) {
        assert.isNull(err);
        assert.equal(res.statusCode, 200);
        assert.deepEqual(
          JSON.parse(body),
          { a: 'foo', b: 'bar' }
        );
      }
    }
  }
}).export(module);

