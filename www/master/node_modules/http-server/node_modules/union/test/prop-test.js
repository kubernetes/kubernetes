var assert = require('assert'),
    request = require('request'),
    vows = require('vows'),
    union = require('../');

vows.describe('union/properties').addBatch({
  'When using `union`': {
    'with a server that responds to requests': {
      topic: function () {
        var callback = this.callback;
        var server = union.createServer({
          before: [
            function (req, res) {
              callback(null, req, res);

              res.writeHead(200, { 'content-type': 'text' });
              res.end();
            }
          ]
        });
        server.listen(9092, function () {
          request('http://localhost:9092/');
        });
      },
      'the `req` should have a proper `httpVersion` set': function (err, req) {
        assert.isNull(err);
        assert.equal(req.httpVersion, '1.1');
      },
      'the `req` should have a proper `httpVersionMajor` set': function (err, req) {
        assert.isNull(err);
        assert.equal(req.httpVersionMajor, 1);
      },
      'the `req` should have a proper `httpVersionMinor` set': function (err, req) {
        assert.isNull(err);
        assert.equal(req.httpVersionMinor, 1);
      },
      'the `req` should have proper `socket` reference set': function (err, req) {
        var net = require('net');

        assert.isNull(err);
        assert.isTrue(req.socket instanceof net.Socket);
      }
    }
  }
}).export(module);
