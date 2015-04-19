/*!
 * socket.io-node
 * Copyright(c) 2011 LearnBoost <dev@learnboost.com>
 * MIT Licensed
 */

/**
 * Module dependencies.
 */

var fs = require('fs')
  , url = require('url')
  , tty = require('tty')
  , crypto = require('crypto')
  , util = require('./util')
  , store = require('./store')
  , client = require('socket.io-client')
  , transports = require('./transports')
  , Logger = require('./logger')
  , Socket = require('./socket')
  , MemoryStore = require('./stores/memory')
  , SocketNamespace = require('./namespace')
  , Static = require('./static')
  , EventEmitter = process.EventEmitter;

/**
 * Export the constructor.
 */

exports = module.exports = Manager;

/**
 * Default transports.
 */

var defaultTransports = exports.defaultTransports = [
    'websocket'
  , 'htmlfile'
  , 'xhr-polling'
  , 'jsonp-polling'
];

/**
 * Inherited defaults.
 */

var parent = module.parent.exports
  , protocol = parent.protocol
  , jsonpolling_re = /^\d+$/;

/**
 * Manager constructor.
 *
 * @param {HTTPServer} server
 * @param {Object} options, optional
 * @api public
 */

function Manager (server, options) {
  this.server = server;
  this.namespaces = {};
  this.sockets = this.of('');
  this.settings = {
      origins: '*:*'
    , log: true
    , store: new MemoryStore
    , logger: new Logger
    , static: new Static(this)
    , heartbeats: true
    , resource: '/socket.io'
    , transports: defaultTransports
    , authorization: false
    , blacklist: ['disconnect']
    , 'log level': 3
    , 'log colors': tty.isatty(process.stdout.fd)
    , 'close timeout': 60
    , 'heartbeat interval': 25
    , 'heartbeat timeout': 60
    , 'polling duration': 20
    , 'flash policy server': true
    , 'flash policy port': 10843
    , 'destroy upgrade': true
    , 'destroy buffer size': 10E7
    , 'browser client': true
    , 'browser client cache': true
    , 'browser client minification': false
    , 'browser client etag': false
    , 'browser client expires': 315360000
    , 'browser client gzip': false
    , 'browser client handler': false
    , 'client store expiration': 15
    , 'match origin protocol': false
  };

  for (var i in options) {
    if (options.hasOwnProperty(i)) {
      this.settings[i] = options[i];
    }
  }

  var self = this;

  // default error handler
  server.on('error', function(err) {
    self.log.warn('error raised: ' + err);
  });

  this.initStore();

  this.on('set:store', function() {
    self.initStore();
  });

  // reset listeners
  this.oldListeners = server.listeners('request').splice(0);
  server.removeAllListeners('request');

  server.on('request', function (req, res) {
    self.handleRequest(req, res);
  });

  server.on('upgrade', function (req, socket, head) {
    self.handleUpgrade(req, socket, head);
  });

  server.on('close', function () {
    clearInterval(self.gc);
  });

  server.once('listening', function () {
    self.gc = setInterval(self.garbageCollection.bind(self), 10000);
  });

  for (var i in transports) {
    if (transports.hasOwnProperty(i)) {
      if (transports[i].init) {
        transports[i].init(this);
      }
    }
  }

  // forward-compatibility with 1.0
  var self = this;
  this.sockets.on('connection', function (conn) {
    self.emit('connection', conn);
  });

  this.sequenceNumber = Date.now() | 0;
 
  this.log.info('socket.io started');
};

Manager.prototype.__proto__ = EventEmitter.prototype

/**
 * Store accessor shortcut.
 *
 * @api public
 */

Manager.prototype.__defineGetter__('store', function () {
  var store = this.get('store');
  store.manager = this;
  return store;
});

/**
 * Logger accessor.
 *
 * @api public
 */

Manager.prototype.__defineGetter__('log', function () {
  var logger = this.get('logger');

  logger.level = this.get('log level') || -1;
  logger.colors = this.get('log colors');
  logger.enabled = this.enabled('log');

  return logger;
});

/**
 * Static accessor.
 *
 * @api public
 */

Manager.prototype.__defineGetter__('static', function () {
  return this.get('static');
});

/**
 * Get settings.
 *
 * @api public
 */

Manager.prototype.get = function (key) {
  return this.settings[key];
};

/**
 * Set settings
 *
 * @api public
 */

Manager.prototype.set = function (key, value) {
  if (arguments.length == 1) return this.get(key);
  this.settings[key] = value;
  this.emit('set:' + key, this.settings[key], key);
  return this;
};

/**
 * Enable a setting
 *
 * @api public
 */

Manager.prototype.enable = function (key) {
  this.settings[key] = true;
  this.emit('set:' + key, this.settings[key], key);
  return this;
};

/**
 * Disable a setting
 *
 * @api public
 */

Manager.prototype.disable = function (key) {
  this.settings[key] = false;
  this.emit('set:' + key, this.settings[key], key);
  return this;
};

/**
 * Checks if a setting is enabled
 *
 * @api public
 */

Manager.prototype.enabled = function (key) {
  return !!this.settings[key];
};

/**
 * Checks if a setting is disabled
 *
 * @api public
 */

Manager.prototype.disabled = function (key) {
  return !this.settings[key];
};

/**
 * Configure callbacks.
 *
 * @api public
 */

Manager.prototype.configure = function (env, fn) {
  if ('function' == typeof env) {
    env.call(this);
  } else if (env == (process.env.NODE_ENV || 'development')) {
    fn.call(this);
  }

  return this;
};

/**
 * Initializes everything related to the message dispatcher.
 *
 * @api private
 */

Manager.prototype.initStore = function () {
  this.handshaken = {};
  this.connected = {};
  this.open = {};
  this.closed = {};
  this.rooms = {};
  this.roomClients = {};

  var self = this;

  this.store.subscribe('handshake', function (id, data) {
    self.onHandshake(id, data);
  });

  this.store.subscribe('connect', function (id) {
    self.onConnect(id);
  });

  this.store.subscribe('open', function (id) {
    self.onOpen(id);
  });

  this.store.subscribe('join', function (id, room) {
    self.onJoin(id, room);
  });

  this.store.subscribe('leave', function (id, room) {
    self.onLeave(id, room);
  });

  this.store.subscribe('close', function (id) {
    self.onClose(id);
  });

  this.store.subscribe('dispatch', function (room, packet, volatile, exceptions) {
    self.onDispatch(room, packet, volatile, exceptions);
  });

  this.store.subscribe('disconnect', function (id) {
    self.onDisconnect(id);
  });

  // we need to do this in a pub/sub way since the client can POST the message
  // over a different socket (ie: different Transport instance)

  //use persistent channel for these, don't add and remove 5 channels for every connection
  //eg. for 10,000 concurrent users this creates 50,000 channels in redis, which kind of slows things down
  //we only need 5 (extra) total channels at all times
  this.store.subscribe('message-remote',function (id, packet) {
    self.onClientMessage(id, packet);
  });

  this.store.subscribe('disconnect-remote', function (id, reason) {
    self.onClientDisconnect(id, reason);
  });

  this.store.subscribe('dispatch-remote', function (id, packet, volatile) {
    var transport = self.transports[id];
    if (transport) {
      transport.onDispatch(packet, volatile);
    }

    if (!volatile) {
      self.onClientDispatch(id, packet);
    }
  });

  this.store.subscribe('heartbeat-clear', function (id) {
    var transport = self.transports[id];
    if (transport) {
      transport.onHeartbeatClear();
    }
  });

  this.store.subscribe('disconnect-force', function (id) {
    var transport = self.transports[id];
    if (transport) {
      transport.onForcedDisconnect();
    }
  });
};
/**
 * Called when a client handshakes.
 *
 * @param text
 */

Manager.prototype.onHandshake = function (id, data) {
  this.handshaken[id] = data;
};

/**
 * Called when a client connects (ie: transport first opens)
 *
 * @api private
 */

Manager.prototype.onConnect = function (id) {
  this.connected[id] = true;
};

/**
 * Called when a client opens a request in a different node.
 *
 * @api private
 */

Manager.prototype.onOpen = function (id) {
  this.open[id] = true;

  if (this.closed[id]) {
    var self = this;

    var transport = self.transports[id];
    if (self.closed[id] && self.closed[id].length && transport) {

      // if we have buffered messages that accumulate between calling
      // onOpen an this async callback, send them if the transport is
      // still open, otherwise leave them buffered
      if (transport.open) {
        transport.payload(self.closed[id]);
        self.closed[id] = [];
      }
    }
  }

  // clear the current transport
  if (this.transports[id]) {
    this.transports[id].discard();
    this.transports[id] = null;
  }
};

/**
 * Called when a message is sent to a namespace and/or room.
 *
 * @api private
 */

Manager.prototype.onDispatch = function (room, packet, volatile, exceptions) {
  if (this.rooms[room]) {
    for (var i = 0, l = this.rooms[room].length; i < l; i++) {
      var id = this.rooms[room][i];

      if (!~exceptions.indexOf(id)) {
        if (this.transports[id] && this.transports[id].open) {
          this.transports[id].onDispatch(packet, volatile);
        } else if (!volatile) {
          this.onClientDispatch(id, packet);
        }
      }
    }
  }
};

/**
 * Called when a client joins a nsp / room.
 *
 * @api private
 */

Manager.prototype.onJoin = function (id, name) {
  if (!this.roomClients[id]) {
    this.roomClients[id] = {};
  }

  if (!this.rooms[name]) {
    this.rooms[name] = [];
  }

  if (!~this.rooms[name].indexOf(id)) {
    this.rooms[name].push(id);
    this.roomClients[id][name] = true;
  }
};

/**
 * Called when a client leaves a nsp / room.
 *
 * @param private
 */

Manager.prototype.onLeave = function (id, room) {
  if (this.rooms[room]) {
    var index = this.rooms[room].indexOf(id);

    if (index >= 0) {
      this.rooms[room].splice(index, 1);
    }

    if (!this.rooms[room].length) {
      delete this.rooms[room];
    }

    if (this.roomClients[id]) {
      delete this.roomClients[id][room];
    }
  }
};

/**
 * Called when a client closes a request in different node.
 *
 * @api private
 */

Manager.prototype.onClose = function (id) {
  if (this.open[id]) {
    delete this.open[id];
  }

  this.closed[id] = [];

  var self = this;
};

/**
 * Dispatches a message for a closed client.
 *
 * @api private
 */

Manager.prototype.onClientDispatch = function (id, packet) {
  if (this.closed[id]) {
    this.closed[id].push(packet);
  }
};

/**
 * Receives a message for a client.
 *
 * @api private
 */

Manager.prototype.onClientMessage = function (id, packet) {
  if (this.namespaces[packet.endpoint]) {
    this.namespaces[packet.endpoint].handlePacket(id, packet);
  }
};

/**
 * Fired when a client disconnects (not triggered).
 *
 * @api private
 */

Manager.prototype.onClientDisconnect = function (id, reason) {
  for (var name in this.namespaces) {
    if (this.namespaces.hasOwnProperty(name)) {
      this.namespaces[name].handleDisconnect(id, reason, typeof this.roomClients[id] !== 'undefined' &&
        typeof this.roomClients[id][name] !== 'undefined');
    }
  }

  this.onDisconnect(id);
};

/**
 * Called when a client disconnects.
 *
 * @param text
 */

Manager.prototype.onDisconnect = function (id) {
  delete this.handshaken[id];

  if (this.open[id]) {
    delete this.open[id];
  }

  if (this.connected[id]) {
    delete this.connected[id];
  }

  if (this.transports[id]) {
    this.transports[id].discard();
    delete this.transports[id];
  }

  if (this.closed[id]) {
    delete this.closed[id];
  }

  if (this.roomClients[id]) {
    for (var room in this.roomClients[id]) {
      if (this.roomClients[id].hasOwnProperty(room)) {
        this.onLeave(id, room);
      }
    }
    delete this.roomClients[id]
  }

  this.store.destroyClient(id, this.get('client store expiration'));
};

/**
 * Handles an HTTP request.
 *
 * @api private
 */

Manager.prototype.handleRequest = function (req, res) {
  var data = this.checkRequest(req);

  if (!data) {
    for (var i = 0, l = this.oldListeners.length; i < l; i++) {
      this.oldListeners[i].call(this.server, req, res);
    }

    return;
  }

  if (data.static || !data.transport && !data.protocol) {
    if (data.static && this.enabled('browser client')) {
      this.static.write(data.path, req, res);
    } else {
      res.writeHead(200);
      res.end('Welcome to socket.io.');

      this.log.info('unhandled socket.io url');
    }

    return;
  }

  if (data.protocol != protocol) {
    res.writeHead(500);
    res.end('Protocol version not supported.');

    this.log.info('client protocol version unsupported');
  } else {
    if (data.id) {
      this.handleHTTPRequest(data, req, res);
    } else {
      this.handleHandshake(data, req, res);
    }
  }
};

/**
 * Handles an HTTP Upgrade.
 *
 * @api private
 */

Manager.prototype.handleUpgrade = function (req, socket, head) {
  var data = this.checkRequest(req)
    , self = this;

  if (!data) {
    if (this.enabled('destroy upgrade')) {
      socket.end();
      this.log.debug('destroying non-socket.io upgrade');
    }

    return;
  }

  req.head = head;
  this.handleClient(data, req);
  req.head = null;
};

/**
 * Handles a normal handshaken HTTP request (eg: long-polling)
 *
 * @api private
 */

Manager.prototype.handleHTTPRequest = function (data, req, res) {
  req.res = res;
  this.handleClient(data, req);
};

/**
 * Intantiantes a new client.
 *
 * @api private
 */

Manager.prototype.handleClient = function (data, req) {
  var socket = req.socket
    , store = this.store
    , self = this;

  // handle sync disconnect xhrs
  if (undefined != data.query.disconnect) {
    if (this.transports[data.id] && this.transports[data.id].open) {
      this.transports[data.id].onForcedDisconnect();
    } else {
      this.store.publish('disconnect-force', data.id);
    }
    req.res.writeHead(200);
    req.res.end();
    return;
  }

  if (!~this.get('transports').indexOf(data.transport)) {
    this.log.warn('unknown transport: "' + data.transport + '"');
    req.connection.end();
    return;
  }

  var transport = new transports[data.transport](this, data, req)
    , handshaken = this.handshaken[data.id];

  if (transport.disconnected) {
    // failed during transport setup
    req.connection.end();
    return;
  }
  if (handshaken) {
    if (transport.open) {
      if (this.closed[data.id] && this.closed[data.id].length) {
        transport.payload(this.closed[data.id]);
        this.closed[data.id] = [];
      }

      this.onOpen(data.id);
      this.store.publish('open', data.id);
      this.transports[data.id] = transport;
    }

    if (!this.connected[data.id]) {
      this.onConnect(data.id);
      this.store.publish('connect', data.id);

      // flag as used
      delete handshaken.issued;
      this.onHandshake(data.id, handshaken);
      this.store.publish('handshake', data.id, handshaken);

      // initialize the socket for all namespaces
      for (var i in this.namespaces) {
        if (this.namespaces.hasOwnProperty(i)) {
          var socket = this.namespaces[i].socket(data.id, true);

          // echo back connect packet and fire connection event
          if (i === '') {
            this.namespaces[i].handlePacket(data.id, { type: 'connect' });
          }
        }
      }
    }
  } else {
    if (transport.open) {
      transport.error('client not handshaken', 'reconnect');
    }

    transport.discard();
  }
};

/**
 * Generates a session id.
 *
 * @api private
 */

Manager.prototype.generateId = function () {
  var rand = new Buffer(15); // multiple of 3 for base64
  if (!rand.writeInt32BE) {
    return Math.abs(Math.random() * Math.random() * Date.now() | 0).toString()
      + Math.abs(Math.random() * Math.random() * Date.now() | 0).toString();
  }
  this.sequenceNumber = (this.sequenceNumber + 1) | 0;
  rand.writeInt32BE(this.sequenceNumber, 11);
  if (crypto.randomBytes) {
    crypto.randomBytes(12).copy(rand);
  } else {
    // not secure for node 0.4
    [0, 4, 8].forEach(function(i) {
      rand.writeInt32BE(Math.random() * Math.pow(2, 32) | 0, i);
    });
  }
  return rand.toString('base64').replace(/\//g, '_').replace(/\+/g, '-');
};

/**
 * Handles a handshake request.
 *
 * @api private
 */

Manager.prototype.handleHandshake = function (data, req, res) {
  var self = this
    , origin = req.headers.origin
    , headers = {
        'Content-Type': 'text/plain'
    };

  function writeErr (status, message) {
    if (data.query.jsonp && jsonpolling_re.test(data.query.jsonp)) {
      res.writeHead(200, { 'Content-Type': 'application/javascript' });
      res.end('io.j[' + data.query.jsonp + '](new Error("' + message + '"));');
    } else {
      res.writeHead(status, headers);
      res.end(message);
    }
  };

  function error (err) {
    writeErr(500, 'handshake error');
    self.log.warn('handshake error ' + err);
  };

  if (!this.verifyOrigin(req)) {
    writeErr(403, 'handshake bad origin');
    return;
  }

  var handshakeData = this.handshakeData(data);

  if (origin) {
    // https://developer.mozilla.org/En/HTTP_Access_Control
    headers['Access-Control-Allow-Origin'] = origin;
    headers['Access-Control-Allow-Credentials'] = 'true';
  }

  this.authorize(handshakeData, function (err, authorized, newData) {
    if (err) return error(err);

    if (authorized) {
      var id = self.generateId()
        , hs = [
              id
            , self.enabled('heartbeats') ? self.get('heartbeat timeout') || '' : ''
            , self.get('close timeout') || ''
            , self.transports(data).join(',')
          ].join(':');

      if (data.query.jsonp && jsonpolling_re.test(data.query.jsonp)) {
        hs = 'io.j[' + data.query.jsonp + '](' + JSON.stringify(hs) + ');';
        res.writeHead(200, { 'Content-Type': 'application/javascript' });
      } else {
        res.writeHead(200, headers);
      }

      self.onHandshake(id, newData || handshakeData);
      self.store.publish('handshake', id, newData || handshakeData);

      res.end(hs);

      self.log.info('handshake authorized', id);
    } else {
      writeErr(403, 'handshake unauthorized');
      self.log.info('handshake unauthorized');
    }
  })
};

/**
 * Gets normalized handshake data
 *
 * @api private
 */

Manager.prototype.handshakeData = function (data) {
  var connection = data.request.connection
    , connectionAddress
    , date = new Date;

  if (connection.remoteAddress) {
    connectionAddress = {
        address: connection.remoteAddress
      , port: connection.remotePort
    };
  } else if (connection.socket && connection.socket.remoteAddress) {
    connectionAddress = {
        address: connection.socket.remoteAddress
      , port: connection.socket.remotePort
    };
  }

  return {
      headers: data.headers
    , address: connectionAddress
    , time: date.toString()
    , query: data.query
    , url: data.request.url
    , xdomain: !!data.request.headers.origin
    , secure: data.request.connection.secure
    , issued: +date
  };
};

/**
 * Verifies the origin of a request.
 *
 * @api private
 */

Manager.prototype.verifyOrigin = function (request) {
  var origin = request.headers.origin || request.headers.referer
    , origins = this.get('origins');

  if (origin === 'null') origin = '*';

  if (origins.indexOf('*:*') !== -1) {
    return true;
  }

  if (origin) {
    try {
      var parts = url.parse(origin);
      parts.port = parts.port || 80;
      var ok =
        ~origins.indexOf(parts.hostname + ':' + parts.port) ||
        ~origins.indexOf(parts.hostname + ':*') ||
        ~origins.indexOf('*:' + parts.port);
      if (!ok) this.log.warn('illegal origin: ' + origin);
      return ok;
    } catch (ex) {
      this.log.warn('error parsing origin');
    }
  }
  else {
    this.log.warn('origin missing from handshake, yet required by config');
  }
  return false;
};

/**
 * Handles an incoming packet.
 *
 * @api private
 */

Manager.prototype.handlePacket = function (sessid, packet) {
  this.of(packet.endpoint || '').handlePacket(sessid, packet);
};

/**
 * Performs authentication.
 *
 * @param Object client request data
 * @api private
 */

Manager.prototype.authorize = function (data, fn) {
  if (this.get('authorization')) {
    var self = this;

    this.get('authorization').call(this, data, function (err, authorized) {
      self.log.debug('client ' + authorized ? 'authorized' : 'unauthorized');
      fn(err, authorized);
    });
  } else {
    this.log.debug('client authorized');
    fn(null, true);
  }

  return this;
};

/**
 * Retrieves the transports adviced to the user.
 *
 * @api private
 */

Manager.prototype.transports = function (data) {
  var transp = this.get('transports')
    , ret = [];

  for (var i = 0, l = transp.length; i < l; i++) {
    var transport = transp[i];

    if (transport) {
      if (!transport.checkClient || transport.checkClient(data)) {
        ret.push(transport);
      }
    }
  }

  return ret;
};

/**
 * Checks whether a request is a socket.io one.
 *
 * @return {Object} a client request data object or `false`
 * @api private
 */

var regexp = /^\/([^\/]+)\/?([^\/]+)?\/?([^\/]+)?\/?$/

Manager.prototype.checkRequest = function (req) {
  var resource = this.get('resource');

  var match;
  if (typeof resource === 'string') {
    match = req.url.substr(0, resource.length);
    if (match !== resource) match = null;
  } else {
    match = resource.exec(req.url);
    if (match) match = match[0];
  }

  if (match) {
    var uri = url.parse(req.url.substr(match.length), true)
      , path = uri.pathname || ''
      , pieces = path.match(regexp);

    // client request data
    var data = {
        query: uri.query || {}
      , headers: req.headers
      , request: req
      , path: path
    };

    if (pieces) {
      data.protocol = Number(pieces[1]);
      data.transport = pieces[2];
      data.id = pieces[3];
      data.static = !!this.static.has(path);
    };

    return data;
  }

  return false;
};

/**
 * Declares a socket namespace
 *
 * @api public
 */

Manager.prototype.of = function (nsp) {
  if (this.namespaces[nsp]) {
    return this.namespaces[nsp];
  }

  return this.namespaces[nsp] = new SocketNamespace(this, nsp);
};

/**
 * Perform garbage collection on long living objects and properties that cannot
 * be removed automatically.
 *
 * @api private
 */

Manager.prototype.garbageCollection = function () {
  // clean up unused handshakes
  var ids = Object.keys(this.handshaken)
    , i = ids.length
    , now = Date.now()
    , handshake;

  while (i--) {
    handshake = this.handshaken[ids[i]];

    if ('issued' in handshake && (now - handshake.issued) >= 3E4) {
      this.onDisconnect(ids[i]);
    }
  }
};
