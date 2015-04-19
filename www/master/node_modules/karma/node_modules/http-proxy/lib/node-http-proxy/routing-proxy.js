/*
 * routing-proxy.js: A routing proxy consuming a RoutingTable and multiple HttpProxy instances
 *
 * (C) 2011 Nodejitsu Inc.
 * MIT LICENCE
 *
 */

var events = require('events'),
    utile = require('utile'),
    HttpProxy = require('./http-proxy').HttpProxy,
    ProxyTable = require('./proxy-table').ProxyTable;

//
// ### function RoutingProxy (options)
// #### @options {Object} Options for this instance
// Constructor function for the RoutingProxy object, a higher level
// reverse proxy Object which can proxy to multiple hosts and also interface
// easily with a RoutingTable instance.
//
var RoutingProxy = exports.RoutingProxy = function (options) {
  events.EventEmitter.call(this);

  var self = this;
  options = options || {};

  if (options.router) {
    this.proxyTable = new ProxyTable(options);
    this.proxyTable.on('routes', function (routes) {
      self.emit('routes', routes);
    });
  }

  //
  // Create a set of `HttpProxy` objects to be used later on calls
  // to `.proxyRequest()` and `.proxyWebSocketRequest()`.
  //
  this.proxies = {};

  //
  // Setup default target options (such as `https`).
  //
  this.target = {};
  this.target.https = options.target && options.target.https;
  this.target.maxSockets = options.target && options.target.maxSockets;

  //
  // Setup other default options to be used for instances of
  // `HttpProxy` created by this `RoutingProxy` instance.
  //
  this.source  = options.source    || { host: 'localhost', port: 8000 };
  this.https   = this.source.https || options.https;
  this.enable  = options.enable;
  this.forward = options.forward;
  this.changeOrigin = options.changeOrigin || false;

  //
  // Listen for 'newListener' events so that we can bind 'proxyError'
  // listeners to each HttpProxy's 'proxyError' event.
  //
  this.on('newListener', function (evt) {
    if (evt === 'proxyError' || evt === 'webSocketProxyError') {
      Object.keys(self.proxies).forEach(function (key) {
        self.proxies[key].on(evt, self.emit.bind(self, evt));
      });
    }
  });
};


//
// Inherit from `events.EventEmitter`.
//
utile.inherits(RoutingProxy, events.EventEmitter);

//
// ### function add (options)
// #### @options {Object} Options for the `HttpProxy` to add.
// Adds a new instance of `HttpProxy` to this `RoutingProxy` instance
// for the specified `options.host` and `options.port`.
//
RoutingProxy.prototype.add = function (options) {
  var self = this,
      key = this._getKey(options);

  //
  // TODO: Consume properties in `options` related to the `ProxyTable`.
  //
  options.target            = options.target       || {};
  options.target.host       = options.target.host  || options.host;
  options.target.port       = options.target.port  || options.port;
  options.target.socketPath = options.target.socketPath || options.socketPath;
  options.target.https      = this.target && this.target.https ||
                              options.target && options.target.https;
  options.target.maxSockets = this.target && this.target.maxSockets;

  //
  // Setup options to pass-thru to the new `HttpProxy` instance
  // for the specified `options.host` and `options.port` pair.
  //
  ['https', 'enable', 'forward', 'changeOrigin'].forEach(function (key) {
    if (options[key] !== false && self[key]) {
      options[key] = self[key];
    }
  });

  this.proxies[key] = new HttpProxy(options);

  if (this.listeners('proxyError').length > 0) {
    this.proxies[key].on('proxyError', this.emit.bind(this, 'proxyError'));
  }

  if (this.listeners('webSocketProxyError').length > 0) {
    this.proxies[key].on('webSocketProxyError', this.emit.bind(this, 'webSocketProxyError'));
  }

  [
    'start',
    'forward',
    'end',
    'proxyResponse',
    'websocket:start',
    'websocket:end',
    'websocket:incoming',
    'websocket:outgoing'
  ].forEach(function (event) {
    this.proxies[key].on(event, this.emit.bind(this, event));
  }, this);
};

//
// ### function remove (options)
// #### @options {Object} Options mapping to the `HttpProxy` to remove.
// Removes an instance of `HttpProxy` from this `RoutingProxy` instance
// for the specified `options.host` and `options.port` (if they exist).
//
RoutingProxy.prototype.remove = function (options) {
  var key = this._getKey(options),
      proxy = this.proxies[key];

  delete this.proxies[key];
  return proxy;
};

//
// ### function close()
// Cleans up any state left behind (sockets, timeouts, etc)
// associated with this instance.
//
RoutingProxy.prototype.close = function () {
  var self = this;

  if (this.proxyTable) {
    //
    // Close the `RoutingTable` associated with
    // this instance (if any).
    //
    this.proxyTable.close();
  }

  //
  // Close all sockets for all `HttpProxy` object(s)
  // associated with this instance.
  //
  Object.keys(this.proxies).forEach(function (key) {
    self.proxies[key].close();
  });
};

//
// ### function proxyRequest (req, res, [port, host, paused])
// #### @req {ServerRequest} Incoming HTTP Request to proxy.
// #### @res {ServerResponse} Outgoing HTTP Request to write proxied data to.
// #### @options {Object} Options for the outgoing proxy request.
//
//     options.port {number} Port to use on the proxy target host.
//     options.host {string} Host of the proxy target.
//     options.buffer {Object} Result from `httpProxy.buffer(req)`
//     options.https {Object|boolean} Settings for https.
//
RoutingProxy.prototype.proxyRequest = function (req, res, options) {
  options = options || {};

  var location;

  //
  // Check the proxy table for this instance to see if we need
  // to get the proxy location for the request supplied. We will
  // always ignore the proxyTable if an explicit `port` and `host`
  // arguments are supplied to `proxyRequest`.
  //
  if (this.proxyTable && !options.host) {
    location = this.proxyTable.getProxyLocation(req);

    //
    // If no location is returned from the ProxyTable instance
    // then respond with `404` since we do not have a valid proxy target.
    //
    if (!location) {
      try {
        if (!this.emit('notFound', req, res)) {
          res.writeHead(404);
          res.end();
        }
      }
      catch (er) {
        console.error("res.writeHead/res.end error: %s", er.message);
      }

      return;
    }

    //
    // When using the ProxyTable in conjunction with an HttpProxy instance
    // only the following arguments are valid:
    //
    // * `proxy.proxyRequest(req, res, { host: 'localhost' })`: This will be skipped
    // * `proxy.proxyRequest(req, res, { buffer: buffer })`: Buffer will get updated appropriately
    // * `proxy.proxyRequest(req, res)`: Options will be assigned appropriately.
    //
    options.port = location.port;
    options.host = location.host;
  }

  var key = this._getKey(options),
      proxy;

  if ((this.target && this.target.https)
    || (location && location.protocol === 'https')) {
    options.target = options.target || {};
    options.target.https = true;
  }

  if (!this.proxies[key]) {
    this.add(utile.clone(options));
  }

  proxy = this.proxies[key];
  proxy.proxyRequest(req, res, options.buffer);
};

//
// ### function proxyWebSocketRequest (req, socket, head, options)
// #### @req {ServerRequest} Websocket request to proxy.
// #### @socket {net.Socket} Socket for the underlying HTTP request
// #### @head {string} Headers for the Websocket request.
// #### @options {Object} Options to use when proxying this request.
//
//     options.port {number} Port to use on the proxy target host.
//     options.host {string} Host of the proxy target.
//     options.buffer {Object} Result from `httpProxy.buffer(req)`
//     options.https {Object|boolean} Settings for https.
//
RoutingProxy.prototype.proxyWebSocketRequest = function (req, socket, head, options) {
  options = options || {};

  var location,
      proxy,
      key;

  if (this.proxyTable && !options.host) {
    location = this.proxyTable.getProxyLocation(req);

    if (!location) {
      return socket.destroy();
    }

    options.port = location.port;
    options.host = location.host;
  }

  key = this._getKey(options);

  if (!this.proxies[key]) {
    this.add(utile.clone(options));
  }

  proxy = this.proxies[key];
  proxy.proxyWebSocketRequest(req, socket, head, options.buffer);
};

//
// ### function addHost (host, target)
// #### @host {String} Host to add to proxyTable
// #### @target {String} Target to add to proxyTable
// Adds a host to proxyTable
//
RoutingProxy.prototype.addHost = function (host, target) {
  if (this.proxyTable) {
    this.proxyTable.addRoute(host, target);
  }
};

//
// ### function removeHost (host)
// #### @host {String} Host to remove from proxyTable
// Removes a host to proxyTable
//
RoutingProxy.prototype.removeHost = function (host) {
  if (this.proxyTable) {
    this.proxyTable.removeRoute(host);
  }
};

//
// ### @private function _getKey (options)
// #### @options {Object} Options to extract the key from
// Ensures that the appropriate options are present in the `options`
// provided and responds with a string key representing the `host`, `port`
// combination contained within.
//
RoutingProxy.prototype._getKey = function (options) {
  if (!options || ((!options.host || !options.port)
    && (!options.target || !options.target.host || !options.target.port))) {
    throw new Error('options.host and options.port or options.target are required.');
  }

  return [
    options.host || options.target.host,
    options.port || options.target.port
  ].join(':');
};
