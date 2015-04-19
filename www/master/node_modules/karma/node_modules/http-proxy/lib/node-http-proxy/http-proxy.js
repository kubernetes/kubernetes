/*
  node-http-proxy.js: http proxy for node.js

  Copyright (c) 2010 Charlie Robbins, Mikeal Rogers, Marak Squires, Fedor Indutny

  Permission is hereby granted, free of charge, to any person obtaining
  a copy of this software and associated documentation files (the
  "Software"), to deal in the Software without restriction, including
  without limitation the rights to use, copy, modify, merge, publish,
  distribute, sublicense, and/or sell copies of the Software, and to
  permit persons to whom the Software is furnished to do so, subject to
  the following conditions:

  The above copyright notice and this permission notice shall be
  included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/

var events = require('events'),
    http = require('http'),
    util = require('util'),
    url = require('url'),
    httpProxy = require('../node-http-proxy');

//
// @private {RegExp} extractPort
// Reusable regular expression for getting the
// port from a host string.
//
var extractPort = /:(\d+)$/;

//
// ### function HttpProxy (options)
// #### @options {Object} Options for this instance.
// Constructor function for new instances of HttpProxy responsible
// for managing the life-cycle of streaming reverse proxyied HTTP requests.
//
// Example options:
//
//      {
//        target: {
//          host: 'localhost',
//          port: 9000
//        },
//        forward: {
//          host: 'localhost',
//          port: 9001
//        }
//      }
//
var HttpProxy = exports.HttpProxy = function (options) {
  if (!options || !options.target) {
    throw new Error('Both `options` and `options.target` are required.');
  }

  events.EventEmitter.call(this);

  var self  = this;

  //
  // Setup basic proxying options:
  //
  // * forward {Object} Options for a forward-proxy (if-any)
  // * target {Object} Options for the **sole** proxy target of this instance
  //
  this.forward  = options.forward;
  this.target   = options.target;
  this.timeout = options.timeout;

  //
  // Setup the necessary instances instance variables for
  // the `target` and `forward` `host:port` combinations
  // used by this instance.
  //
  // * agent {http[s].Agent} Agent to be used by this instance.
  // * protocol {http|https} Core node.js module to make requests with.
  // * base {Object} Base object to create when proxying containing any https settings.
  //
  function setupProxy (key) {
    self[key].agent    = httpProxy._getAgent(self[key]);
    self[key].protocol = httpProxy._getProtocol(self[key]);
    self[key].base     = httpProxy._getBase(self[key]);
  }

  setupProxy('target');
  if (this.forward) {
    setupProxy('forward');
  }

  //
  // Setup opt-in features
  //
  this.enable          = options.enable || {};
  this.enable.xforward = typeof this.enable.xforward === 'boolean'
    ? this.enable.xforward
    : true;

  // if event listener is set then  use it else unlimited.
  this.eventListenerCount = typeof options.eventListenerCount === 'number'? options.eventListenerCount : 0 ; 

  //
  // Setup additional options for WebSocket proxying. When forcing
  // the WebSocket handshake to change the `sec-websocket-location`
  // and `sec-websocket-origin` headers `options.source` **MUST**
  // be provided or the operation will fail with an `origin mismatch`
  // by definition.
  //
  this.source       = options.source       || { host: 'localhost', port: 80 };
  this.source.https = this.source.https    || options.https;
  this.changeOrigin = options.changeOrigin || false;
};

// Inherit from events.EventEmitter
util.inherits(HttpProxy, events.EventEmitter);

//
// ### function proxyRequest (req, res, buffer)
// #### @req {ServerRequest} Incoming HTTP Request to proxy.
// #### @res {ServerResponse} Outgoing HTTP Request to write proxied data to.
// #### @buffer {Object} Result from `httpProxy.buffer(req)`
//
HttpProxy.prototype.proxyRequest = function (req, res, buffer) {
  var self = this,
      errState = false,
      outgoing = new(this.target.base),
      reverseProxy,
      location;

  // If this is a DELETE request then set the "content-length"
  // header (if it is not already set)
  if (req.method === 'DELETE') {
    req.headers['content-length'] = req.headers['content-length'] || '0';
  }

  //
  // Add common proxy headers to the request so that they can
  // be availible to the proxy target server. If the proxy is
  // part of proxy chain it will append the address:
  //
  // * `x-forwarded-for`: IP Address of the original request
  // * `x-forwarded-proto`: Protocol of the original request
  // * `x-forwarded-port`: Port of the original request.
  //
  if (this.enable.xforward && req.connection && req.socket) {
    if (req.headers['x-forwarded-for']) {
      var addressToAppend = "," + req.connection.remoteAddress || req.socket.remoteAddress;
      req.headers['x-forwarded-for'] += addressToAppend;
    }
    else {
      req.headers['x-forwarded-for'] = req.connection.remoteAddress || req.socket.remoteAddress;
    }

    if (req.headers['x-forwarded-port']) {
      var portToAppend = "," + getPortFromHostHeader(req);
      req.headers['x-forwarded-port'] += portToAppend;
    }
    else {
      req.headers['x-forwarded-port'] = getPortFromHostHeader(req);
    }

    if (req.headers['x-forwarded-proto']) {
      var protoToAppend = "," + getProto(req);
      req.headers['x-forwarded-proto'] += protoToAppend;
    }
    else {
      req.headers['x-forwarded-proto'] = getProto(req);
    }
  }

  if (this.timeout) {
    req.socket.setTimeout(this.timeout);
  }

  //
  // Emit the `start` event indicating that we have begun the proxy operation.
  //
  this.emit('start', req, res, this.target);

  //
  // If forwarding is enabled for this instance, foward proxy the
  // specified request to the address provided in `this.forward`
  //
  if (this.forward) {
    this.emit('forward', req, res, this.forward);
    this._forwardRequest(req);
  }

  //
  // #### function proxyError (err)
  // #### @err {Error} Error contacting the proxy target
  // Short-circuits `res` in the event of any error when
  // contacting the proxy target at `host` / `port`.
  //
  function proxyError(err) {
    errState = true;

    //
    // Emit an `error` event, allowing the application to use custom
    // error handling. The error handler should end the response.
    //
    if (self.emit('proxyError', err, req, res)) {
      return;
    }

    res.writeHead(500, { 'Content-Type': 'text/plain' });

    if (req.method !== 'HEAD') {
      //
      // This NODE_ENV=production behavior is mimics Express and
      // Connect.
      //
      if (process.env.NODE_ENV === 'production') {
        res.write('Internal Server Error');
      }
      else {
        res.write('An error has occurred: ' + JSON.stringify(err));
      }
    }

    try { res.end() }
    catch (ex) { console.error("res.end error: %s", ex.message) }
  }

  //
  // Setup outgoing proxy with relevant properties.
  //
  outgoing.host       = this.target.host;
  outgoing.hostname   = this.target.hostname;
  outgoing.port       = this.target.port;
  outgoing.socketPath = this.target.socketPath;
  outgoing.agent      = this.target.agent;
  outgoing.method     = req.method;
  outgoing.path       = url.parse(req.url).path;
  outgoing.headers    = req.headers;

  //
  // If the changeOrigin option is specified, change the
  // origin of the host header to the target URL! Please
  // don't revert this without documenting it!
  //
  if (this.changeOrigin) {
    outgoing.headers.host = this.target.host;
    // Only add port information to the header if not default port
    // for this protocol.
    // See https://github.com/nodejitsu/node-http-proxy/issues/458
    if (this.target.port !== 443 && this.target.https ||
        this.target.port !== 80 && !this.target.https) {
      outgoing.headers.host += ':' + this.target.port;
    }
  }

  //
  // Open new HTTP request to internal resource with will act
  // as a reverse proxy pass
  //
  reverseProxy = this.target.protocol.request(outgoing, function (response) {
    //
    // Process the `reverseProxy` `response` when it's received.
    //
    if (req.httpVersion === '1.0') {
      if (req.headers.connection) {
        response.headers.connection = req.headers.connection
      } else {
        response.headers.connection = 'close'
      }
    } else if (!response.headers.connection) {
      if (req.headers.connection) { response.headers.connection = req.headers.connection }
      else {
        response.headers.connection = 'keep-alive'
      }
    }

    // Remove `Transfer-Encoding` header if client's protocol is HTTP/1.0
    // or if this is a DELETE request with no content-length header.
    // See: https://github.com/nodejitsu/node-http-proxy/pull/373
    if (req.httpVersion === '1.0' || (req.method === 'DELETE'
      && !req.headers['content-length'])) {
      delete response.headers['transfer-encoding'];
    }

    if ((response.statusCode === 301 || response.statusCode === 302)
      && typeof response.headers.location !== 'undefined') {
      location = url.parse(response.headers.location);
      if (location.host === req.headers.host) {
        if (self.source.https && !self.target.https) {
          response.headers.location = response.headers.location.replace(/^http\:/, 'https:');
        }
        if (self.target.https && !self.source.https) {
          response.headers.location = response.headers.location.replace(/^https\:/, 'http:');
        }
      }
    }

    //
    // When the `reverseProxy` `response` ends, end the
    // corresponding outgoing `res` unless we have entered
    // an error state. In which case, assume `res.end()` has
    // already been called and the 'error' event listener
    // removed.
    //
    var ended = false;
    response.on('close', function () {
      if (!ended) { response.emit('end') }
    });

    //
    // After reading a chunked response, the underlying socket
    // will hit EOF and emit a 'end' event, which will abort
    // the request. If the socket was paused at that time,
    // pending data gets discarded, truncating the response.
    // This code makes sure that we flush pending data.
    //
    response.connection.on('end', function () {
      if (response.readable && response.resume) {
        response.resume();
      }
    });

    response.on('end', function () {
      ended = true;
      if (!errState) {
        try { res.end() }
        catch (ex) { console.error("res.end error: %s", ex.message) }

        // Emit the `end` event now that we have completed proxying
        self.emit('end', req, res, response);
      }
    });

    // Allow observer to modify headers or abort response
    try { self.emit('proxyResponse', req, res, response) }
    catch (ex) {
      errState = true;
      return;
    }

    // Set the headers of the client response
    if (res.sentHeaders !== true) {
      Object.keys(response.headers).forEach(function (key) {
        res.setHeader(key, response.headers[key]);
      });
      res.writeHead(response.statusCode);
    }

    function ondata(chunk) {
      if (res.writable) {
        // Only pause if the underlying buffers are full,
        // *and* the connection is not in 'closing' state.
        // Otherwise, the pause will cause pending data to
        // be discarded and silently lost.
        if (false === res.write(chunk) && response.pause
            && response.connection.readable) {
          response.pause();
        }
      }
    }

    response.on('data', ondata);

    function ondrain() {
      if (response.readable && response.resume) {
        response.resume();
      }
    }

    res.on('drain', ondrain);
  });

  // allow unlimited listeners ... 
  reverseProxy.setMaxListeners(this.eventListenerCount);

  //
  // Handle 'error' events from the `reverseProxy`. Setup timeout override if needed
  //
  reverseProxy.once('error', proxyError);

  // Set a timeout on the socket if `this.timeout` is specified.
  reverseProxy.once('socket', function (socket) {
    if (self.timeout) {
      socket.setTimeout(self.timeout);
    }
  });

  //
  // Handle 'error' events from the `req` (e.g. `Parse Error`).
  //
  req.on('error', proxyError);

  //
  // If `req` is aborted, we abort our `reverseProxy` request as well.
  //
  req.on('aborted', function () {
    reverseProxy.abort();
  });

  //
  // For each data `chunk` received from the incoming
  // `req` write it to the `reverseProxy` request.
  //
  req.on('data', function (chunk) {
    if (!errState) {
      var flushed = reverseProxy.write(chunk);
      if (!flushed) {
        req.pause();
        reverseProxy.once('drain', function () {
          try { req.resume() }
          catch (er) { console.error("req.resume error: %s", er.message) }
        });

        //
        // Force the `drain` event in 100ms if it hasn't
        // happened on its own.
        //
        setTimeout(function () {
          reverseProxy.emit('drain');
        }, 100);
      }
    }
  });

  //
  // When the incoming `req` ends, end the corresponding `reverseProxy`
  // request unless we have entered an error state.
  //
  req.on('end', function () {
    if (!errState) {
      reverseProxy.end();
    }
  });

  //Aborts reverseProxy if client aborts the connection.
  req.on('close', function () {
    if (!errState) {
      reverseProxy.abort();
    }
  });

  //
  // If we have been passed buffered data, resume it.
  //
  if (buffer) {
    return !errState
      ? buffer.resume()
      : buffer.destroy();
  }
};

//
// ### function proxyWebSocketRequest (req, socket, head, buffer)
// #### @req {ServerRequest} Websocket request to proxy.
// #### @socket {net.Socket} Socket for the underlying HTTP request
// #### @head {string} Headers for the Websocket request.
// #### @buffer {Object} Result from `httpProxy.buffer(req)`
// Performs a WebSocket proxy operation to the location specified by
// `this.target`.
//
HttpProxy.prototype.proxyWebSocketRequest = function (req, socket, upgradeHead, buffer) {
  var self      = this,
      outgoing  = new(this.target.base),
      listeners = {},
      errState  = false,
      CRLF      = '\r\n',
      //copy upgradeHead to avoid retention of large slab buffers used in node core
      head = new Buffer(upgradeHead.length);
      upgradeHead.copy(head);

  //
  // WebSocket requests must have the `GET` method and
  // the `upgrade:websocket` header
  //
  if (req.method !== 'GET' || req.headers.upgrade.toLowerCase() !== 'websocket') {
    //
    // This request is not WebSocket request
    //
    return socket.destroy();
  }

  //
  // Add common proxy headers to the request so that they can
  // be availible to the proxy target server. If the proxy is
  // part of proxy chain it will append the address:
  //
  // * `x-forwarded-for`: IP Address of the original request
  // * `x-forwarded-proto`: Protocol of the original request
  // * `x-forwarded-port`: Port of the original request.
  //
  if (this.enable.xforward && req.connection) {
    if (req.headers['x-forwarded-for']) {
      var addressToAppend = "," + req.connection.remoteAddress || socket.remoteAddress;
      req.headers['x-forwarded-for'] += addressToAppend;
    }
    else {
      req.headers['x-forwarded-for'] = req.connection.remoteAddress || socket.remoteAddress;
    }

    if (req.headers['x-forwarded-port']) {
      var portToAppend = "," + getPortFromHostHeader(req);
      req.headers['x-forwarded-port'] += portToAppend;
    }
    else {
      req.headers['x-forwarded-port'] = getPortFromHostHeader(req);
    }

    if (req.headers['x-forwarded-proto']) {
      var protoToAppend = "," + (req.connection.pair ? 'wss' : 'ws');
      req.headers['x-forwarded-proto'] += protoToAppend;
    }
    else {
      req.headers['x-forwarded-proto'] = req.connection.pair ? 'wss' : 'ws';
    }
  }

  self.emit('websocket:start', req, socket, head, this.target);

  //
  // Helper function for setting appropriate socket values:
  // 1. Turn of all bufferings
  // 2. For server set KeepAlive
  //
  function _socket(socket, keepAlive) {
    socket.setTimeout(0);
    socket.setNoDelay(true);

    if (keepAlive) {
      if (socket.setKeepAlive) {
        socket.setKeepAlive(true, 0);
      }
      else if (socket.pair.cleartext.socket.setKeepAlive) {
        socket.pair.cleartext.socket.setKeepAlive(true, 0);
      }
    }
  }

  //
  // Setup the incoming client socket.
  //
  _socket(socket, true);

  //
  // On `upgrade` from the Agent socket, listen to
  // the appropriate events.
  //
  function onUpgrade (reverseProxy, proxySocket) {
    if (!reverseProxy) {
      proxySocket.end();
      socket.end();
      return;
    }

    //
    // Any incoming data on this WebSocket to the proxy target
    // will be written to the `reverseProxy` socket.
    //
    proxySocket.on('data', listeners.onIncoming = function (data) {
      if (reverseProxy.incoming.socket.writable) {
        try {
          self.emit('websocket:outgoing', req, socket, head, data);
          var flushed = reverseProxy.incoming.socket.write(data);
          if (!flushed) {
            proxySocket.pause();
            reverseProxy.incoming.socket.once('drain', function () {
              try { proxySocket.resume() }
              catch (er) { console.error("proxySocket.resume error: %s", er.message) }
            });

            //
            // Force the `drain` event in 100ms if it hasn't
            // happened on its own.
            //
            setTimeout(function () {
              reverseProxy.incoming.socket.emit('drain');
            }, 100);
          }
        }
        catch (ex) {
          detach();
        }
      }
    });

    //
    // Any outgoing data on this Websocket from the proxy target
    // will be written to the `proxySocket` socket.
    //
    reverseProxy.incoming.socket.on('data', listeners.onOutgoing = function (data) {
      try {
        self.emit('websocket:incoming', reverseProxy, reverseProxy.incoming, head, data);
        var flushed = proxySocket.write(data);
        if (!flushed) {
          reverseProxy.incoming.socket.pause();
          proxySocket.once('drain', function () {
            try { reverseProxy.incoming.socket.resume() }
            catch (er) { console.error("reverseProxy.incoming.socket.resume error: %s", er.message) }
          });

          //
          // Force the `drain` event in 100ms if it hasn't
          // happened on its own.
          //
          setTimeout(function () {
            proxySocket.emit('drain');
          }, 100);
        }
      }
      catch (ex) {
        detach();
      }
    });

    //
    // Helper function to detach all event listeners
    // from `reverseProxy` and `proxySocket`.
    //
    function detach() {
      proxySocket.destroySoon();
      proxySocket.removeListener('end', listeners.onIncomingClose);
      proxySocket.removeListener('data', listeners.onIncoming);
      reverseProxy.incoming.socket.destroySoon();
      reverseProxy.incoming.socket.removeListener('end', listeners.onOutgoingClose);
      reverseProxy.incoming.socket.removeListener('data', listeners.onOutgoing);
    }

   //
    // If the incoming `proxySocket` socket closes, then
    // detach all event listeners.
    //
    listeners.onIncomingClose = function () {
      reverseProxy.incoming.socket.destroy();
      detach();

      // Emit the `end` event now that we have completed proxying
      self.emit('websocket:end', req, socket, head);
    }

    //
    // If the `reverseProxy` socket closes, then detach all
    // event listeners.
    //
    listeners.onOutgoingClose = function () {
      proxySocket.destroy();
      detach();
    }

    proxySocket.on('end', listeners.onIncomingClose);
    proxySocket.on('close', listeners.onIncomingClose);
    reverseProxy.incoming.socket.on('end', listeners.onOutgoingClose);
    reverseProxy.incoming.socket.on('close', listeners.onOutgoingClose);
  }

  function getPort (port) {
    port = port || 80;
    return port - 80 === 0 ? '' : ':' + port;
  }

  //
  // Get the protocol, and host for this request and create an instance
  // of `http.Agent` or `https.Agent` from the pool managed by `node-http-proxy`.
  //
  var agent        = this.target.agent,
      protocolName = this.target.https ? 'https' : 'http',
      portUri      = getPort(this.source.port),
      remoteHost   = this.target.host + portUri;

  //
  // Change headers (if requested).
  //
  if (this.changeOrigin) {
    req.headers.host   = remoteHost;
    req.headers.origin = protocolName + '://' + remoteHost;
  }

  //
  // Make the outgoing WebSocket request
  //
  outgoing.host    = this.target.host;
  outgoing.port    = this.target.port;
  outgoing.agent   = agent;
  outgoing.method  = 'GET';
  outgoing.path    = req.url;
  outgoing.headers = req.headers;
  outgoing.agent   = agent;

  var reverseProxy = this.target.protocol.request(outgoing);

  //
  // On any errors from the `reverseProxy` emit the
  // `webSocketProxyError` and close the appropriate
  // connections.
  //
  function proxyError (err) {
    reverseProxy.destroy();

    process.nextTick(function () {
      //
      // Destroy the incoming socket in the next tick, in case the error handler
      // wants to write to it.
      //
      socket.destroy();
    });

    self.emit('webSocketProxyError', err, req, socket, head);
  }

  //
  // Here we set the incoming `req`, `socket` and `head` data to the outgoing
  // request so that we can reuse this data later on in the closure scope
  // available to the `upgrade` event. This bookkeeping is not tracked anywhere
  // in nodejs core and is **very** specific to proxying WebSockets.
  //
  reverseProxy.incoming = {
    request: req,
    socket: socket,
    head: head
  };

  //
  // Here we set the handshake `headers` and `statusCode` data to the outgoing
  // request so that we can reuse this data later.
  //
  reverseProxy.handshake = {
    headers: {},
    statusCode: null,
  }

  //
  // If the agent for this particular `host` and `port` combination
  // is not already listening for the `upgrade` event, then do so once.
  // This will force us not to disconnect.
  //
  // In addition, it's important to note the closure scope here. Since
  // there is no mapping of the socket to the request bound to it.
  //
  reverseProxy.on('upgrade', function (res, remoteSocket, head) {
    //
    // Prepare handshake response 'headers' and 'statusCode'.
    //
    reverseProxy.handshake = {
      headers: res.headers,
      statusCode: res.statusCode,
    }

    //
    // Prepare the socket for the reverseProxy request and begin to
    // stream data between the two sockets. Here it is important to
    // note that `remoteSocket._httpMessage === reverseProxy`.
    //
    _socket(remoteSocket, true);
    onUpgrade(remoteSocket._httpMessage, remoteSocket);
  });

  //
  // If the reverseProxy connection has an underlying socket,
  // then execute the WebSocket handshake.
  //
  reverseProxy.once('socket', function (revSocket) {
    revSocket.on('data', function handshake (data) {
      // Set empty headers
      var headers = '';

      //
      // If the handshake statusCode 101, concat headers.
      //
      if (reverseProxy.handshake.statusCode && reverseProxy.handshake.statusCode == 101) {
        headers = [
          'HTTP/1.1 101 Switching Protocols',
          'Upgrade: websocket',
          'Connection: Upgrade',
          'Sec-WebSocket-Accept: ' + reverseProxy.handshake.headers['sec-websocket-accept']
        ];

        headers = headers.concat('', '').join('\r\n');
      }

      //
      // Ok, kind of harmfull part of code. Socket.IO sends a hash
      // at the end of handshake if protocol === 76, but we need
      // to replace 'host' and 'origin' in response so we split
      // data to printable data and to non-printable. (Non-printable
      // will come after double-CRLF).
      //
      var sdata = data.toString();

      // Get the Printable data
      sdata = sdata.substr(0, sdata.search(CRLF + CRLF));

      // Get the Non-Printable data
      data = data.slice(Buffer.byteLength(sdata), data.length);

      if (self.source.https && !self.target.https) {
        //
        // If the proxy server is running HTTPS but the client is running
        // HTTP then replace `ws` with `wss` in the data sent back to the client.
        //
        sdata = sdata.replace('ws:', 'wss:');
      }

      try {
        //
        // Write the printable and non-printable data to the socket
        // from the original incoming request.
        //
        self.emit('websocket:handshake', req, socket, head, sdata, data);
        // add headers to the socket
        socket.write(headers + sdata);
        var flushed = socket.write(data);
        if (!flushed) {
          revSocket.pause();
          socket.once('drain', function () {
            try { revSocket.resume() }
            catch (er) { console.error("reverseProxy.socket.resume error: %s", er.message) }
          });

          //
          // Force the `drain` event in 100ms if it hasn't
          // happened on its own.
          //
          setTimeout(function () {
            socket.emit('drain');
          }, 100);
        }
      }
      catch (ex) {
        //
        // Remove data listener on socket error because the
        // 'handshake' has failed.
        //
        revSocket.removeListener('data', handshake);
        return proxyError(ex);
      }

      //
      // Remove data listener now that the 'handshake' is complete
      //
      revSocket.removeListener('data', handshake);
    });
  });

  //
  // Handle 'error' events from the `reverseProxy`.
  //
  reverseProxy.on('error', proxyError);

  //
  // Handle 'error' events from the `req` (e.g. `Parse Error`).
  //
  req.on('error', proxyError);

  try {
    //
    // Attempt to write the upgrade-head to the reverseProxy
    // request. This is small, and there's only ever one of
    // it; no need for pause/resume.
    //
    // XXX This is very wrong and should be fixed in node's core
    //
    reverseProxy.write(head);
    if (head && head.length === 0) {
      reverseProxy._send('');
    }
  }
  catch (ex) {
    return proxyError(ex);
  }

  //
  // If we have been passed buffered data, resume it.
  //
  if (buffer) {
    return !errState
      ? buffer.resume()
      : buffer.destroy();
  }
};

//
// ### function close()
// Closes all sockets associated with the Agents
// belonging to this instance.
//
HttpProxy.prototype.close = function () {
  [this.forward, this.target].forEach(function (proxy) {
    if (proxy && proxy.agent) {
      for (var host in proxy.agent.sockets) {
        proxy.agent.sockets[host].forEach(function (socket) {
          socket.end();
        });
      }
    }
  });
};

//
// ### @private function _forwardRequest (req)
// #### @req {ServerRequest} Incoming HTTP Request to proxy.
// Forwards the specified `req` to the location specified
// by `this.forward` ignoring errors and the subsequent response.
//
HttpProxy.prototype._forwardRequest = function (req) {
  var self = this,
      outgoing = new(this.forward.base),
      forwardProxy;

  //
  // Setup outgoing proxy with relevant properties.
  //
  outgoing.host    = this.forward.host;
  outgoing.port    = this.forward.port,
  outgoing.agent   = this.forward.agent;
  outgoing.method  = req.method;
  outgoing.path    = req.url;
  outgoing.headers = req.headers;

  //
  // Open new HTTP request to internal resource with will
  // act as a reverse proxy pass.
  //
  forwardProxy = this.forward.protocol.request(outgoing, function (response) {
    //
    // Ignore the response from the forward proxy since this is a 'fire-and-forget' proxy.
    // Remark (indexzero): We will eventually emit a 'forward' event here for performance tuning.
    //
  });

  //
  // Add a listener for the connection timeout event.
  //
  // Remark: Ignoring this error in the event
  //         forward target doesn't exist.
  //
  forwardProxy.once('error', function (err) { });

  //
  // Chunk the client request body as chunks from
  // the proxied request come in
  //
  req.on('data', function (chunk) {
    var flushed = forwardProxy.write(chunk);
    if (!flushed) {
      req.pause();
      forwardProxy.once('drain', function () {
        try { req.resume() }
        catch (er) { console.error("req.resume error: %s", er.message) }
      });

      //
      // Force the `drain` event in 100ms if it hasn't
      // happened on its own.
      //
      setTimeout(function () {
        forwardProxy.emit('drain');
      }, 100);
    }
  });

  //
  // At the end of the client request, we are going to
  // stop the proxied request
  //
  req.on('end', function () {
    forwardProxy.end();
  });
};

function getPortFromHostHeader(req) {
  var match;
  if ((match = extractPort.exec(req.headers.host))) {
    return parseInt(match[1]);
  }

  return getProto(req) === 'https' ? 443 : 80;
}

function getProto(req) {
  return req.isSpdy ? 'https' : (req.connection.pair ? 'https' : 'http');
}
