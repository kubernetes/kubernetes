var test = require('tap').test,
    ecstatic = require('../lib/ecstatic'),
    http = require('http')
;

var server;

test('malformed showdir uri', function (t) {
  server = http.createServer(ecstatic(__dirname, { showDir: true }));
  
  server.listen(0, function () {
    var r = http.get({
      host: 'localhost',
      port: server.address().port,
      path: '/?%'
    });
    r.on('response', function (res) {
      t.equal(res.statusCode, 400);
      t.end();
    });
  });
});

test('server teardown', function (t) {
  server.close();

  var to = setTimeout(function () {
    process.stderr.write('# server not closing; slaughtering process.\n');
    process.exit(0);
  }, 5000);
  t.end();
});
