var fs = require('fs');
var http = require('http');

http.createServer(function(req, res) {
  switch (req.url) {
    case '/':
    case '/index.html': {
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(fs.readFileSync('index.html', 'utf8'));
      break;
    }

    case '/browser-source-map-support.js': {
      res.writeHead(200, { 'Content-Type': 'text/javascript' });
      res.end(fs.readFileSync('../browser-source-map-support.js', 'utf8'));
      break;
    }

    case '/script.js': {
      res.writeHead(200, { 'Content-Type': 'text/javascript', 'SourceMap': 'script-source-map.map' });
      res.end(fs.readFileSync('script.js', 'utf8'));
      break;
    }

    case '/script-source-map.map': {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(fs.readFileSync('script.map', 'utf8'));
      break;
    }

    case '/header-test/script.coffee': {
      res.writeHead(200, { 'Content-Type': 'text/x-coffeescript' });
      res.end(fs.readFileSync('script.coffee', 'utf8'));
      break;
    }

    default: {
      res.writeHead(404, { 'Content-Type': 'text/html' });
      res.end('404 not found');
      break;
    }
  }
}).listen(1337, '127.0.0.1');

console.log('Server running at http://127.0.0.1:1337/');
