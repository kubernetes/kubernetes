var http = require('http')
  , fspfs = require('../');

var server = http.createServer(function(q,r){ r.writeHead(200); r.end(':3') }) 
  , flash = fspfs.createServer();

server.listen(8080);
flash.listen(8081,server);