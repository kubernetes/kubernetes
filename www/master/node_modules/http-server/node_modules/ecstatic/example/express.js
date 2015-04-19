var express = require('express');
var ecstatic = require('../lib/ecstatic');
var http = require('http');

var app = express();
app.use(ecstatic({
  root: __dirname + '/public',
  showdir : true
}));
http.createServer(app).listen(8080);

console.log('Listening on :8080');
