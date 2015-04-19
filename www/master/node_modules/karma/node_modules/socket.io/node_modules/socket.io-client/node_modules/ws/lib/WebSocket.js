/*!
 * ws: a node.js websocket client
 * Copyright(c) 2011 Einar Otto Stangvik <einaros@gmail.com>
 * MIT Licensed
 */

var util = require('util')
  , events = require('events')
  , http = require('http')
  , https = require('https')
  , crypto = require('crypto')
  , url = require('url')
  , stream = require('stream')
  , Options = require('options')
  , Sender = require('./Sender')
  , Receiver = require('./Receiver')
  , SenderHixie = require('./Sender.hixie')
  , ReceiverHixie = require('./Receiver.hixie');

/**
 * Constants
 */

// Default protocol version

var protocolVersion = 13;

// Close timeout

var closeTimeout = 30000; // Allow 5 seconds to terminate the connection cleanly

/**
 * WebSocket implementation
 */

function WebSocket(address, protocols, options) {

  if (protocols && !Array.isArray(protocols) && 'object' == typeof protocols) {
    // accept the "options" Object as the 2nd argument
    options = protocols;
    protocols = null;
  }
  if ('string' == typeof protocols) {
    protocols = [ protocols ];
  }
  if (!Array.isArray(protocols)) {
    protocols = [];
  }
  // TODO: actually handle the `Sub-Protocols` part of the WebSocket client

  this._socket = null;
  this.bytesReceived = 0;
  this.readyState = null;
  this.supports = {};

  if (Array.isArray(address)) {
    initAsServerClient.apply(this, address.concat(options));
  } else {
    initAsClient.apply(this, [address, protocols, options]);
  }
}

/**
 * Inherits from EventEmitter.
 */

util.inherits(WebSocket, events.EventEmitter);

/**
 * Ready States
 */

["CONNECTING", "OPEN", "CLOSING", "CLOSED"].forEach(function (state, index) {
    WebSocket.prototype[state] = WebSocket[state] = index;
});

/**
 * Gracefully closes the connection, after sending a description message to the server
 *
 * @param {Object} data to be sent to the server
 * @api public
 */

WebSocket.prototype.close = function(code, data) {
  if (this.readyState == WebSocket.CLOSING || this.readyState == WebSocket.CLOSED) return;
  if (this.readyState == WebSocket.CONNECTING) {
    this.readyState = WebSocket.CLOSED;
    return;
  }
  try {
    this.readyState = WebSocket.CLOSING;
    this._closeCode = code;
    this._closeMessage = data;
    var mask = !this._isServer;
    this._sender.close(code, data, mask);
  }
  catch (e) {
    this.emit('error', e);
  }
  finally {
    this.terminate();
  }
}

/**
 * Pause the client stream
 *
 * @api public
 */

WebSocket.prototype.pause = function() {
  if (this.readyState != WebSocket.OPEN) throw new Error('not opened');
  return this._socket.pause();
}

/**
 * Sends a ping
 *
 * @param {Object} data to be sent to the server
 * @param {Object} Members - mask: boolean, binary: boolean
 * @param {boolean} dontFailWhenClosed indicates whether or not to throw if the connection isnt open
 * @api public
 */

WebSocket.prototype.ping = function(data, options, dontFailWhenClosed) {
  if (this.readyState != WebSocket.OPEN) {
    if (dontFailWhenClosed === true) return;
    throw new Error('not opened');
  }
  options = options || {};
  if (typeof options.mask == 'undefined') options.mask = !this._isServer;
  this._sender.ping(data, options);
}

/**
 * Sends a pong
 *
 * @param {Object} data to be sent to the server
 * @param {Object} Members - mask: boolean, binary: boolean
 * @param {boolean} dontFailWhenClosed indicates whether or not to throw if the connection isnt open
 * @api public
 */

WebSocket.prototype.pong = function(data, options, dontFailWhenClosed) {
  if (this.readyState != WebSocket.OPEN) {
    if (dontFailWhenClosed === true) return;
    throw new Error('not opened');
  }
  options = options || {};
  if (typeof options.mask == 'undefined') options.mask = !this._isServer;
  this._sender.pong(data, options);
}

/**
 * Resume the client stream
 *
 * @api public
 */

WebSocket.prototype.resume = function() {
  if (this.readyState != WebSocket.OPEN) throw new Error('not opened');
  return this._socket.resume();
}

/**
 * Sends a piece of data
 *
 * @param {Object} data to be sent to the server
 * @param {Object} Members - mask: boolean, binary: boolean
 * @param {function} Optional callback which is executed after the send completes
 * @api public
 */

WebSocket.prototype.send = function(data, options, cb) {
  if (typeof options == 'function') {
    cb = options;
    options = {};
  }
  if (this.readyState != WebSocket.OPEN) {
    if (typeof cb == 'function') cb(new Error('not opened'));
    else throw new Error('not opened');
    return;
  }
  if (!data) data = '';
  if (this._queue) {
    var self = this;
    this._queue.push(function() { self.send(data, options, cb); });
    return;
  }
  options = options || {};
  options.fin = true;
  if (typeof options.binary == 'undefined') {
    options.binary = (data instanceof ArrayBuffer || data instanceof Buffer ||
      data instanceof Uint8Array ||
      data instanceof Uint16Array ||
      data instanceof Uint32Array ||
      data instanceof Int8Array ||
      data instanceof Int16Array ||
      data instanceof Int32Array ||
      data instanceof Float32Array ||
      data instanceof Float64Array);
  }
  if (typeof options.mask == 'undefined') options.mask = !this._isServer;
  var readable = typeof stream.Readable == 'function' ? stream.Readable : stream.Stream;
  if (data instanceof readable) {
    startQueue(this);
    var self = this;
    sendStream(this, data, options, function(error) {
      process.nextTick(function() { executeQueueSends(self); });
      if (typeof cb == 'function') cb(error);
    });
  }
  else this._sender.send(data, options, cb);
}

/**
 * Streams data through calls to a user supplied function
 *
 * @param {Object} Members - mask: boolean, binary: boolean
 * @param {function} 'function (error, send)' which is executed on successive ticks of which send is 'function (data, final)'.
 * @api public
 */

WebSocket.prototype.stream = function(options, cb) {
  if (typeof options == 'function') {
    cb = options;
    options = {};
  }
  var self = this;
  if (typeof cb != 'function') throw new Error('callback must be provided');
  if (this.readyState != WebSocket.OPEN) {
    if (typeof cb == 'function') cb(new Error('not opened'));
    else throw new Error('not opened');
    return;
  }
  if (this._queue) {
    this._queue.push(function() { self.stream(options, cb); });
    return;
  }
  options = options || {};
  if (typeof options.mask == 'undefined') options.mask = !this._isServer;
  startQueue(this);
  var send = function(data, final) {
    try {
      if (self.readyState != WebSocket.OPEN) throw new Error('not opened');
      options.fin = final === true;
      self._sender.send(data, options);
      if (!final) process.nextTick(cb.bind(null, null, send));
      else executeQueueSends(self);
    }
    catch (e) {
      if (typeof cb == 'function') cb(e);
      else {
        delete self._queue;
        self.emit('error', e);
      }
    }
  }
  process.nextTick(cb.bind(null, null, send));
}

/**
 * Immediately shuts down the connection
 *
 * @api public
 */

WebSocket.prototype.terminate = function() {
  if (this.readyState == WebSocket.CLOSED) return;
  if (this._socket) {
    try {
      // End the connection
      this._socket.end();
    }
    catch (e) {
      // Socket error during end() call, so just destroy it right now
      cleanupWebsocketResources.call(this, true);
      return;
    }

    // Add a timeout to ensure that the connection is completely
    // cleaned up within 30 seconds, even if the clean close procedure
    // fails for whatever reason
    this._closeTimer = setTimeout(cleanupWebsocketResources.bind(this, true), closeTimeout);
  }
  else if (this.readyState == WebSocket.CONNECTING) {
    cleanupWebsocketResources.call(this, true);
  }
};

/**
 * Expose bufferedAmount
 *
 * @api public
 */

Object.defineProperty(WebSocket.prototype, 'bufferedAmount', {
  get: function get() {
    var amount = 0;
    if (this._socket) {
      amount = this._socket.bufferSize || 0;
    }
    return amount;
  }
});

/**
 * Emulates the W3C Browser based WebSocket interface using function members.
 *
 * @see http://dev.w3.org/html5/websockets/#the-websocket-interface
 * @api public
 */

['open', 'error', 'close', 'message'].forEach(function(method) {
  Object.defineProperty(WebSocket.prototype, 'on' + method, {
    /**
     * Returns the current listener
     *
     * @returns {Mixed} the set function or undefined
     * @api public
     */

    get: function get() {
      var listener = this.listeners(method)[0];
      return listener ? (listener._listener ? listener._listener : listener) : undefined;
    },

    /**
     * Start listening for events
     *
     * @param {Function} listener the listener
     * @returns {Mixed} the set function or undefined
     * @api public
     */

    set: function set(listener) {
      this.removeAllListeners(method);
      this.addEventListener(method, listener);
    }
  });
});

/**
 * Emulates the W3C Browser based WebSocket interface using addEventListener.
 *
 * @see https://developer.mozilla.org/en/DOM/element.addEventListener
 * @see http://dev.w3.org/html5/websockets/#the-websocket-interface
 * @api public
 */
WebSocket.prototype.addEventListener = function(method, listener) {
  var target = this;
  if (typeof listener === 'function') {
    if (method === 'message') {
      function onMessage (data, flags) {
        listener.call(this, new MessageEvent(data, flags.binary ? 'Binary' : 'Text', target));
      }
      // store a reference so we can return the original function from the addEventListener hook
      onMessage._listener = listener;
      this.on(method, onMessage);
    } else if (method === 'close') {
      function onClose (code, message) {
        listener.call(this, new CloseEvent(code, message, target));
      }
      // store a reference so we can return the original function from the addEventListener hook
      onClose._listener = listener;
      this.on(method, onClose);
    } else if (method === 'error') {
      function onError (event) {
        event.target = target;
        listener.call(this, event);
      }
      // store a reference so we can return the original function from the addEventListener hook
      onError._listener = listener;
      this.on(method, onError);
    } else if (method === 'open') {
      function onOpen () {
        listener.call(this, new OpenEvent(target));
      }
      // store a reference so we can return the original function from the addEventListener hook
      onOpen._listener = listener;
      this.on(method, onOpen);
    } else {
      this.on(method, listener);
    }
  }
}

module.exports = WebSocket;

/**
 * W3C MessageEvent
 *
 * @see http://www.w3.org/TR/html5/comms.html
 * @api private
 */

function MessageEvent(dataArg, typeArg, target) {
  this.data = dataArg;
  this.type = typeArg;
  this.target = target;
}

/**
 * W3C CloseEvent
 *
 * @see http://www.w3.org/TR/html5/comms.html
 * @api private
 */

function CloseEvent(code, reason, target) {
  this.wasClean = (typeof code == 'undefined' || code == 1000);
  this.code = code;
  this.reason = reason;
  this.target = target;
}

/**
 * W3C OpenEvent
 *
 * @see http://www.w3.org/TR/html5/comms.html
 * @api private
 */

function OpenEvent(target) {
  this.target = target;
}

/**
 * Entirely private apis,
 * which may or may not be bound to a sepcific WebSocket instance.
 */

function initAsServerClient(req, socket, upgradeHead, options) {
  options = new Options({
    protocolVersion: protocolVersion,
    protocol: null
  }).merge(options);

  // expose state properties
  this.protocol = options.value.protocol;
  this.protocolVersion = options.value.protocolVersion;
  this.supports.binary = (this.protocolVersion != 'hixie-76');
  this.upgradeReq = req;
  this.readyState = WebSocket.CONNECTING;
  this._isServer = true;

  // establish connection
  if (options.value.protocolVersion == 'hixie-76') establishConnection.call(this, ReceiverHixie, SenderHixie, socket, upgradeHead);
  else establishConnection.call(this, Receiver, Sender, socket, upgradeHead);
}

function initAsClient(address, protocols, options) {
  options = new Options({
    origin: null,
    protocolVersion: protocolVersion,
    host: null,
    headers: null,
    protocol: null,
    agent: null,

    // ssl-related options
    pfx: null,
    key: null,
    passphrase: null,
    cert: null,
    ca: null,
    ciphers: null,
    rejectUnauthorized: null
  }).merge(options);
  if (options.value.protocolVersion != 8 && options.value.protocolVersion != 13) {
    throw new Error('unsupported protocol version');
  }

  // verify url and establish http class
  var serverUrl = url.parse(address);
  var isUnixSocket = serverUrl.protocol === 'ws+unix:';
  if (!serverUrl.host && !isUnixSocket) throw new Error('invalid url');
  var isSecure = serverUrl.protocol === 'wss:' || serverUrl.protocol === 'https:';
  var httpObj = isSecure ? https : http;
  var port = serverUrl.port || (isSecure ? 443 : 80);
  var auth = serverUrl.auth;

  // expose state properties
  this._isServer = false;
  this.url = address;
  this.protocolVersion = options.value.protocolVersion;
  this.supports.binary = (this.protocolVersion != 'hixie-76');

  // begin handshake
  var key = new Buffer(options.value.protocolVersion + '-' + Date.now()).toString('base64');
  var shasum = crypto.createHash('sha1');
  shasum.update(key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11');
  var expectedServerKey = shasum.digest('base64');

  var agent = options.value.agent;

  var headerHost = serverUrl.hostname;
  // Append port number to Host and Origin header, only if specified in the url and non-default
  if(serverUrl.port) {
    if((isSecure && (port != 443)) || (!isSecure && (port != 80))){
      headerHost = headerHost + ':' + port;
    }
  }

  var requestOptions = {
    port: port,
    host: serverUrl.hostname,
    headers: {
      'Connection': 'Upgrade',
      'Upgrade': 'websocket',
      'Host': headerHost,
      'Origin': headerHost,
      'Sec-WebSocket-Version': options.value.protocolVersion,
      'Sec-WebSocket-Key': key
    }
  };

  // If we have basic auth.
  if (auth) {
    requestOptions.headers['Authorization'] = 'Basic ' + new Buffer(auth).toString('base64');
  }

  if (options.value.protocol) {
    requestOptions.headers['Sec-WebSocket-Protocol'] = options.value.protocol;
  }

  if (options.value.host) {
    requestOptions.headers['Host'] = options.value.host;
  }

  if (options.value.headers) {
    for (var header in options.value.headers) {
       if (options.value.headers.hasOwnProperty(header)) {
        requestOptions.headers[header] = options.value.headers[header];
       }
    }
  }

  if (options.isDefinedAndNonNull('pfx')
   || options.isDefinedAndNonNull('key')
   || options.isDefinedAndNonNull('passphrase')
   || options.isDefinedAndNonNull('cert')
   || options.isDefinedAndNonNull('ca')
   || options.isDefinedAndNonNull('ciphers')
   || options.isDefinedAndNonNull('rejectUnauthorized')) {

    if (options.isDefinedAndNonNull('pfx')) requestOptions.pfx = options.value.pfx;
    if (options.isDefinedAndNonNull('key')) requestOptions.key = options.value.key;
    if (options.isDefinedAndNonNull('passphrase')) requestOptions.passphrase = options.value.passphrase;
    if (options.isDefinedAndNonNull('cert')) requestOptions.cert = options.value.cert;
    if (options.isDefinedAndNonNull('ca')) requestOptions.ca = options.value.ca;
    if (options.isDefinedAndNonNull('ciphers')) requestOptions.ciphers = options.value.ciphers;
    if (options.isDefinedAndNonNull('rejectUnauthorized')) requestOptions.rejectUnauthorized = options.value.rejectUnauthorized;

    if (!agent) {
        // global agent ignores client side certificates
        agent = new httpObj.Agent(requestOptions);
    }
  }

  requestOptions.path = serverUrl.path || '/';

  if (agent) {
    requestOptions.agent = agent;
  }

  if (isUnixSocket) {
    requestOptions.socketPath = serverUrl.pathname;
  }
  if (options.value.origin) {
    if (options.value.protocolVersion < 13) requestOptions.headers['Sec-WebSocket-Origin'] = options.value.origin;
    else requestOptions.headers['Origin'] = options.value.origin;
  }

  var self = this;
  var req = httpObj.request(requestOptions);

  req.on('error', function(error) {
    self.emit('error', error);
    cleanupWebsocketResources.call(this, error);
  });

  req.once('response', function(res) {
    if (!self.emit('unexpected-response', req, res)) {
      var error = new Error('unexpected server response (' + res.statusCode + ')');
      req.abort();
      self.emit('error', error);
    }
    cleanupWebsocketResources.call(this, error);
  });

  req.once('upgrade', function(res, socket, upgradeHead) {
    if (self.readyState == WebSocket.CLOSED) {
      // client closed before server accepted connection
      self.emit('close');
      self.removeAllListeners();
      socket.end();
      return;
    }
    var serverKey = res.headers['sec-websocket-accept'];
    if (typeof serverKey == 'undefined' || serverKey !== expectedServerKey) {
      self.emit('error', 'invalid server key');
      self.removeAllListeners();
      socket.end();
      return;
    }

    var serverProt = res.headers['sec-websocket-protocol'];
    var protList = (options.value.protocol || "").split(/, */);
    var protError = null;
    if (!options.value.protocol && serverProt) {
        protError = 'server sent a subprotocol even though none requested';
    } else if (options.value.protocol && !serverProt) {
        protError = 'server sent no subprotocol even though requested';
    } else if (serverProt && protList.indexOf(serverProt) === -1) {
        protError = 'server responded with an invalid protocol';
    }
    if (protError) {
        self.emit('error', protError);
        self.removeAllListeners();
        socket.end();
        return;
    } else if (serverProt) {
        self.protocol = serverProt;
    }

    establishConnection.call(self, Receiver, Sender, socket, upgradeHead);

    // perform cleanup on http resources
    req.removeAllListeners();
    req = null;
    agent = null;
  });

  req.end();
  this.readyState = WebSocket.CONNECTING;
}

function establishConnection(ReceiverClass, SenderClass, socket, upgradeHead) {
  this._socket = socket;
  socket.setTimeout(0);
  socket.setNoDelay(true);
  var self = this;
  this._receiver = new ReceiverClass();

  // socket cleanup handlers
  socket.on('end', cleanupWebsocketResources.bind(this));
  socket.on('close', cleanupWebsocketResources.bind(this));
  socket.on('error', cleanupWebsocketResources.bind(this));

  // ensure that the upgradeHead is added to the receiver
  function firstHandler(data) {
    if (self.readyState != WebSocket.OPEN) return;
    if (upgradeHead && upgradeHead.length > 0) {
      self.bytesReceived += upgradeHead.length;
      var head = upgradeHead;
      upgradeHead = null;
      self._receiver.add(head);
    }
    dataHandler = realHandler;
    if (data) {
      self.bytesReceived += data.length;
      self._receiver.add(data);
    }
  }
  // subsequent packets are pushed straight to the receiver
  function realHandler(data) {
    if (data) self.bytesReceived += data.length;
    self._receiver.add(data);
  }
  var dataHandler = firstHandler;
  // if data was passed along with the http upgrade,
  // this will schedule a push of that on to the receiver.
  // this has to be done on next tick, since the caller
  // hasn't had a chance to set event handlers on this client
  // object yet.
  process.nextTick(firstHandler);

  // receiver event handlers
  self._receiver.ontext = function (data, flags) {
    flags = flags || {};
    self.emit('message', data, flags);
  };
  self._receiver.onbinary = function (data, flags) {
    flags = flags || {};
    flags.binary = true;
    self.emit('message', data, flags);
  };
  self._receiver.onping = function(data, flags) {
    flags = flags || {};
    self.pong(data, {mask: !self._isServer, binary: flags.binary === true}, true);
    self.emit('ping', data, flags);
  };
  self._receiver.onpong = function(data, flags) {
    self.emit('pong', data, flags);
  };
  self._receiver.onclose = function(code, data, flags) {
    flags = flags || {};
    self.close(code, data);
  };
  self._receiver.onerror = function(reason, errorCode) {
    // close the connection when the receiver reports a HyBi error code
    self.close(typeof errorCode != 'undefined' ? errorCode : 1002, '');
    self.emit('error', reason, errorCode);
  };

  // finalize the client
  this._sender = new SenderClass(socket);
  this._sender.on('error', function(error) {
    self.close(1002, '');
    self.emit('error', error);
  });
  this.readyState = WebSocket.OPEN;
  this.emit('open');

  socket.on('data', dataHandler);
}

function startQueue(instance) {
  instance._queue = instance._queue || [];
}

function executeQueueSends(instance) {
  var queue = instance._queue;
  if (typeof queue == 'undefined') return;
  delete instance._queue;
  for (var i = 0, l = queue.length; i < l; ++i) {
    queue[i]();
  }
}

function sendStream(instance, stream, options, cb) {
  stream.on('data', function(data) {
    if (instance.readyState != WebSocket.OPEN) {
      if (typeof cb == 'function') cb(new Error('not opened'));
      else {
        delete instance._queue;
        instance.emit('error', new Error('not opened'));
      }
      return;
    }
    options.fin = false;
    instance._sender.send(data, options);
  });
  stream.on('end', function() {
    if (instance.readyState != WebSocket.OPEN) {
      if (typeof cb == 'function') cb(new Error('not opened'));
      else {
        delete instance._queue;
        instance.emit('error', new Error('not opened'));
      }
      return;
    }
    options.fin = true;
    instance._sender.send(null, options);
    if (typeof cb == 'function') cb(null);
  });
}

function cleanupWebsocketResources(error) {
  if (this.readyState == WebSocket.CLOSED) return;
  var emitClose = this.readyState != WebSocket.CONNECTING;
  this.readyState = WebSocket.CLOSED;

  clearTimeout(this._closeTimer);
  this._closeTimer = null;
  if (emitClose) this.emit('close', this._closeCode || 1000, this._closeMessage || '');

  if (this._socket) {
    this._socket.removeAllListeners();
    // catch all socket error after removing all standard handlers
    var socket = this._socket;
    this._socket.on('error', function() {
      try { socket.destroy(); } catch (e) {}
    });
    try {
      if (!error) this._socket.end();
      else this._socket.destroy();
    }
    catch (e) { /* Ignore termination errors */ }
    this._socket = null;
  }
  if (this._sender) {
    this._sender.removeAllListeners();
    this._sender = null;
  }
  if (this._receiver) {
    this._receiver.cleanup();
    this._receiver = null;
  }
  this.removeAllListeners();
  this.on('error', function() {}); // catch all errors after this
  delete this._queue;
}
