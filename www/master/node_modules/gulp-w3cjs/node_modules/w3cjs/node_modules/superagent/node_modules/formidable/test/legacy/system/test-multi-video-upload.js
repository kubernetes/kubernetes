var common = require('../common');
var BOUNDARY = '---------------------------10102754414578508781458777923',
    FIXTURE = TEST_FIXTURES+'/multi_video.upload',
    fs = require('fs'),
    http = require('http'),
    formidable = require(common.lib + '/index'),
    server = http.createServer();

server.on('request', function(req, res) {
  var form = new formidable.IncomingForm(),
      uploads = {};

  form.uploadDir = TEST_TMP;
  form.hash = 'sha1';
  form.parse(req);

  form
    .on('fileBegin', function(field, file) {
      assert.equal(field, 'upload');

      var tracker = {file: file, progress: [], ended: false};
      uploads[file.name] = tracker;
      file
        .on('progress', function(bytesReceived) {
          tracker.progress.push(bytesReceived);
          assert.equal(bytesReceived, file.size);
        })
        .on('end', function() {
          tracker.ended = true;
        });
    })
    .on('field', function(field, value) {
      assert.equal(field, 'title');
      assert.equal(value, '');
    })
    .on('file', function(field, file) {
      assert.equal(field, 'upload');
      assert.strictEqual(uploads[file.name].file, file);
    })
    .on('end', function() {
      assert.ok(uploads['shortest_video.flv']);
      assert.ok(uploads['shortest_video.flv'].ended);
      assert.ok(uploads['shortest_video.flv'].progress.length > 3);
      assert.equal(uploads['shortest_video.flv'].file.hash, 'd6a17616c7143d1b1438ceeef6836d1a09186b3a');
      assert.equal(uploads['shortest_video.flv'].progress.slice(-1), uploads['shortest_video.flv'].file.size);
      assert.ok(uploads['shortest_video.mp4']);
      assert.ok(uploads['shortest_video.mp4'].ended);
      assert.ok(uploads['shortest_video.mp4'].progress.length > 3);
      assert.equal(uploads['shortest_video.mp4'].file.hash, '937dfd4db263f4887ceae19341dcc8d63bcd557f');

      server.close();
      res.writeHead(200);
      res.end('good');
    });
});

server.listen(TEST_PORT, function() {
  var stat, headers, request, fixture;

  stat = fs.statSync(FIXTURE);
  request = http.request({
    port: TEST_PORT,
    path: '/',
    method: 'POST',
    headers: {
      'content-type': 'multipart/form-data; boundary='+BOUNDARY,
      'content-length': stat.size,
    },
  });
  fs.createReadStream(FIXTURE).pipe(request);
});
