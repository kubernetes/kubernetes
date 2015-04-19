var assert = require('assert'),
    fs = require('fs'),
    path = require('path'),
    request = require('request'),
    vows = require('vows'),
    union = require('../');

vows.describe('union/streaming').addBatch({
  'When using `union`': {
    'a simple union server': {
      topic: function () {
        var self = this;

        union.createServer({
          buffer: false,
          before: [
            function (req, res, next) {
              var chunks = '';

              req.on('data', function (chunk) {
                chunks += chunk;
              });

              req.on('end', function () {
                self.callback(null, chunks);
              });
            }
          ]
        }).listen(9000, function () {
          request.post('http://localhost:9000').write('hello world');
        });
      },
      'should receive complete POST data': function (chunks) {
        assert.equal(chunks, 'hello world');
      }
    },
    "a simple pipe to a file": {
      topic: function () {
        var self = this;

        union.createServer({
          before: [
            function (req, res, next) {
              var filename = path.join(__dirname, 'fixtures', 'pipe-write-test.txt'),
                  writeStream = fs.createWriteStream(filename);

              req.pipe(writeStream);
              writeStream.on('close', function () {
                res.writeHead(200);
                fs.createReadStream(filename).pipe(res);
              });
            }
          ]
        }).listen(9044, function () {
          request({
            method: 'POST',
            uri: 'http://localhost:9044',
            body: 'hello world'
          }, self.callback);
        });
      },
      'should receive complete POST data': function (err, res, body) {
        assert.equal(body, 'hello world');
      }
    }
  }
}).export(module);

