var test = require('tap').test,
    ecstatic = require('../lib/ecstatic'),
    http = require('http'),
    express = require('express'),
    request = require('request'),
    mkdirp = require('mkdirp'),
    fs = require('fs'),
    path = require('path');

var root = __dirname + '/public',
    baseDir = 'base';

mkdirp.sync(root + '/emptyDir');

var cases = require('./common-cases-error');

test('express', function (t) {
  var filenames = Object.keys(cases);
  var port = Math.floor(Math.random() * ((1<<16) - 1e4) + 1e4);

  var app = express();

  app.use(ecstatic({
    root: root,
    gzip: true,
    baseDir: baseDir,
    autoIndex: true,
    showDir: true,
    cache: "no-cache",
    handleError: false
  }));

  var server = http.createServer(app);

  server.listen(port, function () {
    var pending = filenames.length;
    filenames.forEach(function (file) {
      var uri = 'http://localhost:' + port + path.join('/', baseDir, file),
          headers = cases[file].headers || {};

      request.get({
        uri: uri,
        followRedirect: false,
        headers: headers
      }, function (err, res, body) {
        if (err) t.fail(err);
        var r = cases[file];
        t.equal(res.statusCode, r.code, 'status code for `' + file + '`');

        if (r.code === 200) {
            t.equal(res.headers['cache-control'], 'no-cache', 'cache control for `' + file + '`');
        };

        if (r.type !== undefined) {
          t.equal(
            res.headers['content-type'].split(';')[0], r.type,
            'content-type for `' + file + '`'
          );
        }

        if (r.body !== undefined) {
          t.equal(body, r.body, 'body for `' + file + '`');
        }

        if (--pending === 0) {
          server.close();
          t.end();
        }
      });
    });
  });
});
