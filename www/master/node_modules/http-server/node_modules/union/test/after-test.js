var assert = require('assert'),
    vows = require('vows'),
    request = require('request'),
    union = require('../');

function stream_callback(cb) {
  return function () {
    var stream   = new union.ResponseStream();

    stream.once("pipe", function (req) {
      return cb ? cb(null,req) : undefined;
    });

    return stream;
  };
}

vows.describe('union/after').addBatch({
  'When using `union`': {
    'a union server with after middleware': {
      topic: function () {
        var self = this;

        union.createServer({
          after: [ stream_callback(), stream_callback(self.callback) ]
        }).listen(9000, function () {
          request.get('http://localhost:9000');
        });
      },
      'should preserve the request until the last call': function (req) {
        assert.equal(req.req.httpVersion, '1.1');
        assert.equal(req.req.url, '/');
        assert.equal(req.req.method, 'GET');
      }
    }
  }
}).export(module);