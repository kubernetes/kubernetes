var http = require('http');
var ecstatic = require('../lib/ecstatic')({
  root: __dirname + '/public',
  showDir: true,
  autoIndex: true
});

http.createServer(ecstatic).listen(8080);

console.log('Listening on :8080');
