var test = require('tap').test,
    ecstatic = require('../lib/ecstatic'),
    http = require('http'),
    request = require('request'),
    mkdirp = require('mkdirp'),
    fs = require('fs'),
    path = require('path');

var root = __dirname + '/public',
    baseDir = 'base';

test('304_not_modified', function (t) {
  var port = Math.floor(Math.random() * ((1<<16) - 1e4) + 1e4),
      file = 'a.txt';
  
  var server = http.createServer(
    ecstatic({
      root: root,
      gzip: true,
      baseDir: baseDir,
      autoIndex: true,
      showDir: true
    })
  );

  server.listen(port, function () {
    var uri = 'http://localhost:' + port + path.join('/', baseDir, file),
        now = (new Date()).toString();

    request.get({
      uri: uri,
      followRedirect: false,
    }, function (err, res, body) {
      if (err) t.fail(err);

      t.equal(res.statusCode, 200, 'first request should be a 200');

      request.get({
        uri: uri,
        followRedirect: false,
        headers: { 'if-modified-since': now }
      }, function (err, res, body) {
        if (err) t.fail(err);

        t.equal(res.statusCode, 304, 'second request should be a 304');
        server.close();
        t.end();
      });
    });
  });
});
