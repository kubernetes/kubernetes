/*!
 * socket.io-node
 * Copyright(c) 2011 LearnBoost <dev@learnboost.com>
 * MIT Licensed
 */

/**
 * Module requirements.
 */

var Transport = require('../../transport')
  , EventEmitter = process.EventEmitter
  , crypto = require('crypto')
  , parser = require('../../parser');

/**
 * Export the constructor.
 */

exports = module.exports = WebSocket;

/**
 * HTTP interface constructor. Interface compatible with all transports that
 * depend on request-response cycles.
 *
 * @api public
 */

function WebSocket (mng, data, req) {
  // parser
  var self = this;

  this.parser = new Parser({maxBuffer: mng.get('destroy buffer size')});
  this.parser.on('data', function (packet) {
    self.log.debug(self.name + ' received data packet', packet);
    self.onMessage(parser.decodePacket(packet));
  });
  this.parser.on('close', function () {
    self.end();
  });
  this.parser.on('error', function () {
    self.end();
  });
  this.parser.on('kick', function (reason) {
    self.log.warn(self.name + ' parser forced user kick: ' + reason);
    self.onMessage({type: 'disconnect', endpoint: ''});
    self.end();
  });

  Transport.call(this, mng, data, req);
};

/**
 * Inherits from Transport.
 */

WebSocket.prototype.__proto__ = Transport.prototype;

/**
 * Transport name
 *
 * @api public
 */

WebSocket.prototype.name = 'websocket';

/**
 * Websocket draft version
 *
 * @api public
 */

WebSocket.prototype.protocolVersion = 'hixie-76';

/**
 * Called when the socket connects.
 *
 * @api private
 */

WebSocket.prototype.onSocketConnect = function () {
  var self = this;

  this.socket.setNoDelay(true);

  this.buffer = true;
  this.buffered = [];

  if (this.req.headers.upgrade !== 'WebSocket') {
    this.log.warn(this.name + ' connection invalid');
    this.end();
    return;
  }

  var origin = this.req.headers['origin']
  , waitingForNonce = false;
  if(this.manager.settings['match origin protocol']){
    location = (origin.indexOf('https')>-1 ? 'wss' : 'ws') + '://' + this.req.headers.host + this.req.url;
  }else if(this.socket.encrypted){
    location = 'wss://' + this.req.headers.host + this.req.url;
  }else{
    location = 'ws://' + this.req.headers.host + this.req.url;
  }

  if (this.req.headers['sec-websocket-key1']) {
    // If we don't have the nonce yet, wait for it (HAProxy compatibility).
    if (! (this.req.head && this.req.head.length >= 8)) {
      waitingForNonce = true;
    }

    var headers = [
        'HTTP/1.1 101 WebSocket Protocol Handshake'
      , 'Upgrade: WebSocket'
      , 'Connection: Upgrade'
      , 'Sec-WebSocket-Origin: ' + origin
      , 'Sec-WebSocket-Location: ' + location
    ];

    if (this.req.headers['sec-websocket-protocol']){
      headers.push('Sec-WebSocket-Protocol: '
          + this.req.headers['sec-websocket-protocol']);
    }
  } else {
    var headers = [
        'HTTP/1.1 101 Web Socket Protocol Handshake'
      , 'Upgrade: WebSocket'
      , 'Connection: Upgrade'
      , 'WebSocket-Origin: ' + origin
      , 'WebSocket-Location: ' + location
    ];
  }

  try {
    this.socket.write(headers.concat('', '').join('\r\n'));
    this.socket.setTimeout(0);
    this.socket.setNoDelay(true);
    this.socket.setEncoding('utf8');
  } catch (e) {
    this.end();
    return;
  }

  if (waitingForNonce) {
    this.socket.setEncoding('binary');
  } else if (this.proveReception(headers)) {
    self.flush();
  }

  var headBuffer = '';

  this.socket.on('data', function (data) {
    if (waitingForNonce) {
      headBuffer += data;

      if (headBuffer.length < 8) {
        return;
      }

      // Restore the connection to utf8 encoding after receiving the nonce
      self.socket.setEncoding('utf8');
      waitingForNonce = false;

      // Stuff the nonce into the location where it's expected to be
      self.req.head = headBuffer.substr(0, 8);
      headBuffer = '';

      if (self.proveReception(headers)) {
        self.flush();
      }

      return;
    }

    self.parser.add(data);
  });
};

/**
 * Writes to the socket.
 *
 * @api private
 */

WebSocket.prototype.write = function (data) {
  if (this.open) {
    this.drained = false;

    if (this.buffer) {
      this.buffered.push(data);
      return this;
    }

    var length = Buffer.byteLength(data)
      , buffer = new Buffer(2 + length);

    buffer.write('\x00', 'binary');
    buffer.write(data, 1, 'utf8');
    buffer.write('\xff', 1 + length, 'binary');

    try {
      if (this.socket.write(buffer)) {
        this.drained = true;
      }
    } catch (e) {
      this.end();
    }

    this.log.debug(this.name + ' writing', data);
  }
};

/**
 * Flushes the internal buffer
 *
 * @api private
 */

WebSocket.prototype.flush = function () {
  this.buffer = false;

  for (var i = 0, l = this.buffered.length; i < l; i++) {
    this.write(this.buffered.splice(0, 1)[0]);
  }
};

/**
 * Finishes the handshake.
 *
 * @api private
 */

WebSocket.prototype.proveReception = function (headers) {
  var self = this
    , k1 = this.req.headers['sec-websocket-key1']
    , k2 = this.req.headers['sec-websocket-key2'];

  if (k1 && k2){
    var md5 = crypto.createHash('md5');

    [k1, k2].forEach(function (k) {
      var n = parseInt(k.replace(/[^\d]/g, ''))
        , spaces = k.replace(/[^ ]/g, '').length;

      if (spaces === 0 || n % spaces !== 0){
        self.log.warn('Invalid ' + self.name + ' key: "' + k + '".');
        self.end();
        return false;
      }

      n /= spaces;

      md5.update(String.fromCharCode(
        n >> 24 & 0xFF,
        n >> 16 & 0xFF,
        n >> 8  & 0xFF,
        n       & 0xFF));
    });

    md5.update(this.req.head.toString('binary'));

    try {
      this.socket.write(md5.digest('binary'), 'binary');
    } catch (e) {
      this.end();
    }
  }

  return true;
};

/**
 * Writes a payload.
 *
 * @api private
 */

WebSocket.prototype.payload = function (msgs) {
  for (var i = 0, l = msgs.length; i < l; i++) {
    this.write(msgs[i]);
  }

  return this;
};

/**
 * Closes the connection.
 *
 * @api private
 */

WebSocket.prototype.doClose = function () {
  this.socket.end();
};

/**
 * WebSocket parser
 *
 * @api public
 */

function Parser (opts) {
  this._maxBuffer = (opts && opts.maxBuffer) || 10E7;
  this._dataLength = 0;
  this.buffer = '';
  this.i = 0;
};

/**
 * Inherits from EventEmitter.
 */

Parser.prototype.__proto__ = EventEmitter.prototype;

/**
 * Adds data to the buffer.
 *
 * @api public
 */

Parser.prototype.add = function (data) {
  this._dataLength += data.length;
  if(this._dataLength > this._maxBuffer) {
    this.buffer = ''; //Clear buffer
    this.emit('kick', 'max buffer size reached');
    return;
  }

  this.buffer += data;
  this.parse();
};

/**
 * Parses the buffer.
 *
 * @api private
 */

Parser.prototype.parse = function () {
  for (var i = this.i, chr, l = this.buffer.length; i < l; i++){
    chr = this.buffer[i];

    if (this.buffer.length == 2 && this.buffer[1] == '\u0000') {
      this.emit('close');
      this.buffer = '';
      this.i = 0;
      return;
    }

    if (i === 0){
      if (chr != '\u0000')
        this.error('Bad framing. Expected null byte as first frame');
      else
        continue;
    }

    if (chr == '\ufffd'){
      this.emit('data', this.buffer.substr(1, i - 1));
      this.buffer = this.buffer.substr(i + 1);
      this.i = 0;
      return this.parse();
    }
  }
};

/**
 * Handles an error
 *
 * @api private
 */

Parser.prototype.error = function (reason) {
  this.buffer = '';
  this.i = 0;
  this.emit('error', reason);
  return this;
};
