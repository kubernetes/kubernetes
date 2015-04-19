# rainbowsocks

SOCKS4a client developed with rainbows

## Install

`npm install rainbowsocks`

## API

### var rainbowsocks = new RainbowSocks(port, [host])

* port - Socks4a Proxy Port
* host - SOCKS4a Proxy Host || 127.0.0.1

#### rainbowsocks.connect(targetHost, targetPort, callback)

Pseudo function of rainbowsocks.request to establish a TCP/IP stream connection

* targetHost - IP/Domain of desired destination
* targetPort - Port of desired destination
* callback - Called with signature of (err, socket)

#### rainbowsocks.bind(targetHost, targetPort, callback)

Pseudo function of rainbowsocks.request to establish a TCP/IP port binding

* targetHost - IP/Domain of desired destination
* targetPort - Port of desired destination
* callback - Called with signature of (err, socket)

#### rainbowsocks.request(cmdBuf, domain, port, callback)

Sends a request to proxy to take a specific action

* cmdBuf - 1 Octet Buffer containg SOCKS4a action code
* targetHost - IP/Domain of desired destination
* targetPort - Port of desired destination
* callback - Called with signature of (err, socket)

#### Event: `connect`

Connected to proxy


## Example

```javascript
var RainbowSocks = require('rainbowsocks');
var sock = new RainbowSocks(8080, '192.168.0.45');

sock.on('connect', function() {
  console.log('Connected to proxy');
  sock.connect('www.google.com', 80, function(err, socket) {
    if(err) throw err;
    console.log('Connected to www.google.com');
    socket.write('GET / HTTP/1.1\nHost: www.google.com\n\n');
    socket.pipe(process.stdout);
  });
});
```

## Licence

MIT