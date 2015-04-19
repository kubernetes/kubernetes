var assert = require('assert'),
    request = require('request'),
    vows = require('vows'),
    union = require('../');

vows.describe('union/header').addBatch({
  'When using `union`': {
    'with a server that responds with a header': {
      topic: function () {
        var callback = this.callback;
        var server = union.createServer({
          before: [
            function (req, res) {
              res.on('header', function () {
                callback(null, res);
              });
              res.writeHead(200, { 'content-type': 'text' });
              res.end();
            }
          ]
        });
        server.listen(9092, function () {
          request('http://localhost:9092/');
        });
      },
      'it should have proper `headerSent` set': function (err, res) {
        assert.isNull(err);
        assert.isTrue(res.headerSent);
      },
      'it should have proper `_emittedHeader` set': function (err, res) {
        assert.isNull(err);
        assert.isTrue(res._emittedHeader);
      }
    }
  }
}).export(module);
