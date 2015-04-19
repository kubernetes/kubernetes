var http = require('http'),
    httpProxy = require('../../lib/node-http-proxy');

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

//
// Create a HttpProxy object for each target
//

var proxies = addresses.map(function (target) {
  return new httpProxy.HttpProxy({
    target: target
  });
});

//
// Get the proxy at the front of the array, put it at the end and return it
// If you want a fancier balancer, put your code here
//

function nextProxy() {
  var proxy = proxies.shift();
  proxies.push(proxy);
  return proxy;
}

// 
// Get the 'next' proxy and send the http request 
//

var server = http.createServer(function (req, res) {    
  nextProxy().proxyRequest(req, res);
});

// 
// Get the 'next' proxy and send the upgrade request 
//

server.on('upgrade', function (req, socket, head) {
  nextProxy().proxyWebSocketRequest(req, socket, head);
});

server.listen(8080);  
  