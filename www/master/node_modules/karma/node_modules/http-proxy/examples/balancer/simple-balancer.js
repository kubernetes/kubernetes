var httpProxy = require('../../lib/node-http-proxy');
//
// A simple round-robin load balancing strategy.
// 
// First, list the servers you want to use in your rotation.
//
var addresses = [
  {
    host: 'ws1.0.0.0',
    port: 80
  },
  {
    host: 'ws2.0.0.0',
    port: 80
  }
];

httpProxy.createServer(function (req, res, proxy) {
  //
  // On each request, get the first location from the list...
  //
  var target = addresses.shift();

  //
  // ...then proxy to the server whose 'turn' it is...
  //
  console.log('balancing request to: ', target);
  proxy.proxyRequest(req, res, target);

  //
  // ...and then the server you just used becomes the last item in the list.
  //
  addresses.push(target);
}).listen(8000);

// Rinse; repeat; enjoy.