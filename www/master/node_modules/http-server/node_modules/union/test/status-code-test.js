var assert = require('assert'),
    request = require('request'),
    vows = require('vows'),
    union = require('../');

vows.describe('union/status-code').addBatch({
  'When using `union`': {
    'with a server setting `res.statusCode`': {
      topic: function () {
        var server = union.createServer({
          before: [
            function (req, res) {
              res.statusCode = 404;
              res.end();
            }
          ]
        });
        server.listen(9091, this.callback);
      },
      'and sending a request': {
        topic: function () {
          request('http://localhost:9091/', this.callback);
        },
        'it should have proper `statusCode` set': function (err, res, body) {
          assert.isTrue(!err);
          assert.equal(res.statusCode, 404);
        }
      }
    }
  }
}).export(module);
