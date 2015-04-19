var util = require('util'),
    colors = require('colors'),
    http = require('http'),
    httpProxy = require('../../lib/node-http-proxy'),
    Store = require('../helpers/store') 

http.createServer(new Store().handler()).listen(7531)

// Now we set up our proxy.
httpProxy.createServer(
  // This is where our middlewares go, with any options desired - in this case,
  // the list of routes/URLs and their destinations.
  require('proxy-by-url')({
    '/store': { port: 7531, host: 'localhost' },
    '/': { port: 9000, host: 'localhost' }
  })
).listen(8000);

//
// Target Http Server (to listen for requests on 'localhost')
//
http.createServer(function (req, res) {
  res.writeHead(200, { 'Content-Type': 'text/plain' });
  res.write('request successfully proxied to: ' + req.url + '\n' + JSON.stringify(req.headers, true, 2));
  res.end();
}).listen(9000);

// And finally, some colored startup output.
util.puts('http proxy server'.blue + ' started '.green.bold + 'on port '.blue + '8000'.yellow);
util.puts('http server '.blue + 'started '.green.bold + 'on port '.blue + '9000 '.yellow);
