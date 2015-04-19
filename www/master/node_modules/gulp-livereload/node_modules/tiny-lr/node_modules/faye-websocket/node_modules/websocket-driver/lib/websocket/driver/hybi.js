'use strict';

var crypto     = require('crypto'),
    util       = require('util'),
    Extensions = require('websocket-extensions'),
    Base       = require('./base'),
    Frame      = require('./hybi/frame'),
    Message    = require('./hybi/message'),
    Reader     = require('./hybi/stream_reader');

var Hybi = function(request, url, options) {
  Base.apply(this, arguments);

  this._extensions     = new Extensions();
  this._reader         = new Reader();
  this._stage          = 0;
  this._masking        = this._options.masking;
  this._protocols      = this._options.protocols || [];
  this._requireMasking = this._options.requireMasking;
  this._pingCallbacks  = {};

  if (typeof this._protocols === 'string')
    this._protocols = this._protocols.split(/ *, */);

  if (!this._request) return;

  var secKey    = this._request.headers['sec-websocket-key'],
      protos    = this._request.headers['sec-websocket-protocol'],
      version   = this._request.headers['sec-websocket-version'],
      supported = this._protocols;

  this._headers.set('Upgrade', 'websocket');
  this._headers.set('Connection', 'Upgrade');
  this._headers.set('Sec-WebSocket-Accept', Hybi.generateAccept(secKey));

  if (protos !== undefined) {
    if (typeof protos === 'string') protos = protos.split(/ *, */);
    this.protocol = protos.filter(function(p) { return supported.indexOf(p) >= 0 })[0];
    if (this.protocol) this._headers.set('Sec-WebSocket-Protocol', this.protocol);
  }

  this.version = 'hybi-' + version;
};
util.inherits(Hybi, Base);

Hybi.mask = function(payload, mask, offset) {
  if (!mask || mask.length === 0) return payload;
  offset = offset || 0;

  for (var i = 0, n = payload.length - offset; i < n; i++) {
    payload[offset + i] = payload[offset + i] ^ mask[i % 4];
  }
  return payload;
};

Hybi.generateAccept = function(key) {
  var sha1 = crypto.createHash('sha1');
  sha1.update(key + Hybi.GUID);
  return sha1.digest('base64');
};

Hybi.GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';

var instance = {
  BYTE:   255,
  FIN:    128,
  MASK:   128,
  RSV1:   64,
  RSV2:   32,
  RSV3:   16,
  OPCODE: 15,
  LENGTH: 127,

  OPCODES: {
    continuation: 0,
    text:         1,
    binary:       2,
    close:        8,
    ping:         9,
    pong:         10
  },

  OPCODE_CODES:    [0, 1, 2, 8, 9, 10],
  MESSAGE_OPCODES: [0, 1, 2],
  OPENING_OPCODES: [1, 2],

  TWO_POWERS: [0, 1, 2, 3, 4, 5, 6, 7].map(function(n) { return Math.pow(2, 8 * n) }),

  ERRORS: {
    normal_closure:       1000,
    going_away:           1001,
    protocol_error:       1002,
    unacceptable:         1003,
    encoding_error:       1007,
    policy_violation:     1008,
    too_large:            1009,
    extension_error:      1010,
    unexpected_condition: 1011
  },

  ERROR_CODES:        [1000, 1001, 1002, 1003, 1007, 1008, 1009, 1010, 1011],
  MIN_RESERVED_ERROR: 3000,
  MAX_RESERVED_ERROR: 4999,

  // http://www.w3.org/International/questions/qa-forms-utf-8.en.php
  UTF8_MATCH: /^([\x00-\x7F]|[\xC2-\xDF][\x80-\xBF]|\xE0[\xA0-\xBF][\x80-\xBF]|[\xE1-\xEC\xEE\xEF][\x80-\xBF]{2}|\xED[\x80-\x9F][\x80-\xBF]|\xF0[\x90-\xBF][\x80-\xBF]{2}|[\xF1-\xF3][\x80-\xBF]{3}|\xF4[\x80-\x8F][\x80-\xBF]{2})*$/,

  addExtension: function(extension) {
    this._extensions.add(extension);
    return true;
  },

  parse: function(data) {
    this._reader.put(data);
    var buffer = true;
    while (buffer) {
      switch (this._stage) {
        case 0:
          buffer = this._reader.read(1);
          if (buffer) this._parseOpcode(buffer[0]);
          break;

        case 1:
          buffer = this._reader.read(1);
          if (buffer) this._parseLength(buffer[0]);
          break;

        case 2:
          buffer = this._reader.read(this._frame.lengthBytes);
          if (buffer) this._parseExtendedLength(buffer);
          break;

        case 3:
          buffer = this._reader.read(4);
          if (buffer) {
            this._frame.maskingKey = buffer;
            this._stage = 4;
          }
          break;

        case 4:
          buffer = this._reader.read(this._frame.length);
          if (buffer) {
            this._emitFrame(buffer);
            this._stage = 0;
          }
          break;

        default:
          buffer = null;
      }
    }
  },

  text: function(message) {
    if (this.readyState > 1) return false;
    return this.frame(message, 'text');
  },

  binary: function(message) {
    if (this.readyState > 1) return false;
    return this.frame(message, 'binary');
  },

  ping: function(message, callback) {
    if (this.readyState > 1) return false;
    message = message || '';
    if (callback) this._pingCallbacks[message] = callback;
    return this.frame(message, 'ping');
  },

  close: function(reason, code) {
    reason = reason || '';
    code   = code   || this.ERRORS.normal_closure;

    if (this.readyState <= 0) {
      this.readyState = 3;
      this.emit('close', new Base.CloseEvent(code, reason));
      return true;
    } else if (this.readyState === 1) {
      this.readyState = 2;
      this._extensions.close(function() { this.frame(reason, 'close', code) }, this);
      return true;
    } else {
      return false;
    }
  },

  frame: function(data, type, code) {
    if (this.readyState <= 0) return this._queue([data, type, code]);
    if (this.readyState > 2) return false;

    if (data instanceof Array) data = new Buffer(data);

    var message = new Message(),
        isText  = (typeof data === 'string'),
        payload, buffer;

    message.rsv1   = message.rsv2 = message.rsv3 = false;
    message.opcode = this.OPCODES[type || (isText ? 'text' : 'binary')];

    payload = isText ? new Buffer(data, 'utf8') : data;

    if (code) {
      buffer = payload;
      payload = new Buffer(2 + buffer.length);
      payload[0] = ~~(code / 256) & this.BYTE;
      payload[1] = code & this.BYTE;
      buffer.copy(payload, 2);
    }
    message.data = payload;

    var onMessageReady = function(message) {
      var frame = new Frame();

      frame.final   = true;
      frame.rsv1    = message.rsv1;
      frame.rsv2    = message.rsv2;
      frame.rsv3    = message.rsv3;
      frame.opcode  = message.opcode;
      frame.masked  = !!this._masking;
      frame.length  = message.data.length;
      frame.payload = message.data;

      if (frame.masked) frame.maskingKey = crypto.randomBytes(4);

      this._sendFrame(frame);
    };

    if (this.MESSAGE_OPCODES.indexOf(message.opcode) >= 0)
      this._extensions.processOutgoingMessage(message, function(error, message) {
        if (error) return this._fail('extension_error', error.message);
        onMessageReady.call(this, message);
      }, this);
    else
      onMessageReady.call(this, message);

    return true;
  },

  _sendFrame: function(frame) {
    var length = frame.length,
        header = (length <= 125) ? 2 : (length <= 65535 ? 4 : 10),
        offset = header + (frame.masked ? 4 : 0),
        buffer = new Buffer(offset + length),
        BYTE   = this.BYTE,
        masked = frame.masked ? this.MASK : 0;

    buffer[0] = (frame.final ? this.FIN : 0) |
                (frame.rsv1 ? this.RSV1 : 0) |
                (frame.rsv2 ? this.RSV2 : 0) |
                (frame.rsv3 ? this.RSV3 : 0) |
                frame.opcode;

    if (length <= 125) {
      buffer[1] = masked | length;
    } else if (length <= 65535) {
      buffer[1] = masked | 126;
      buffer[2] = ~~(length / 256);
      buffer[3] = length & BYTE;
    } else {
      buffer[1] = masked | 127;
      buffer[2] = ~~(length / Math.pow(2, 56)) & BYTE;
      buffer[3] = ~~(length / Math.pow(2, 48)) & BYTE;
      buffer[4] = ~~(length / Math.pow(2, 40)) & BYTE;
      buffer[5] = ~~(length / Math.pow(2, 32)) & BYTE;
      buffer[6] = ~~(length / Math.pow(2, 24)) & BYTE;
      buffer[7] = ~~(length / Math.pow(2, 16)) & BYTE;
      buffer[8] = ~~(length / Math.pow(2, 8))  & BYTE;
      buffer[9] = length & BYTE;
    }

    if (frame.masked) {
      frame.maskingKey.copy(buffer, header);
      Hybi.mask(frame.payload, frame.maskingKey).copy(buffer, offset);
    } else {
      frame.payload.copy(buffer, offset);
    }

    this._write(buffer);
  },

  _handshakeResponse: function() {
    try {
      var extensions = this._extensions.generateResponse(this._request.headers['sec-websocket-extensions']);
    } catch (e) {
      return this._fail('protocol_error', e.message);
    }

    if (extensions) this._headers.set('Sec-WebSocket-Extensions', extensions);

    var start   = 'HTTP/1.1 101 Switching Protocols',
        headers = [start, this._headers.toString(), ''];

    return new Buffer(headers.join('\r\n'), 'utf8');
  },

  _shutdown: function(code, reason) {
    delete this._frame;
    delete this._message;

    var sendCloseFrame = (this.readyState === 1);
    this.readyState = 2;
    this._stage = 5;

    this._extensions.close(function() {
      if (sendCloseFrame) this.frame(reason, 'close', code);
      this.readyState = 3;
      this.emit('close', new Base.CloseEvent(code, reason));
    }, this);
  },

  _fail: function(type, message) {
    if (this.readyState > 1) return;
    this.emit('error', new Error(message));
    this._shutdown(this.ERRORS[type], message);
  },

  _parseOpcode: function(data) {
    var rsvs = [this.RSV1, this.RSV2, this.RSV3].map(function(rsv) {
      return (data & rsv) === rsv;
    });

    var frame = this._frame = new Frame();

    frame.final  = (data & this.FIN) === this.FIN;
    frame.rsv1   = rsvs[0];
    frame.rsv2   = rsvs[1];
    frame.rsv3   = rsvs[2];
    frame.opcode = (data & this.OPCODE);

    if (!this._extensions.validFrameRsv(frame))
      return this._fail('protocol_error',
          'One or more reserved bits are on: reserved1 = ' + (frame.rsv1 ? 1 : 0) +
          ', reserved2 = ' + (frame.rsv2 ? 1 : 0) +
          ', reserved3 = ' + (frame.rsv3 ? 1 : 0));

    if (this.OPCODE_CODES.indexOf(frame.opcode) < 0)
      return this._fail('protocol_error', 'Unrecognized frame opcode: ' + frame.opcode);

    if (this.MESSAGE_OPCODES.indexOf(frame.opcode) < 0 && !frame.final)
      return this._fail('protocol_error', 'Received fragmented control frame: opcode = ' + frame.opcode);

    if (this._message && this.OPENING_OPCODES.indexOf(frame.opcode) >= 0)
      return this._fail('protocol_error', 'Received new data frame but previous continuous frame is unfinished');

    this._stage = 1;
  },

  _parseLength: function(data) {
    var frame = this._frame;

    frame.masked = (data & this.MASK) === this.MASK;
    if (this._requireMasking && !frame.masked)
      return this._fail('unacceptable', 'Received unmasked frame but masking is required');

    frame.length = (data & this.LENGTH);

    if (frame.length >= 0 && frame.length <= 125) {
      if (!this._checkFrameLength()) return;
      this._stage = frame.masked ? 3 : 4;
    } else {
      frame.lengthBytes = (frame.length === 126 ? 2 : 8);
      this._stage = 2;
    }
  },

  _parseExtendedLength: function(buffer) {
    var frame = this._frame;
    frame.length = this._getInteger(buffer);

    if (this.MESSAGE_OPCODES.indexOf(frame.opcode) < 0 && frame.length > 125)
      return this._fail('protocol_error', 'Received control frame having too long payload: ' + frame.length);

    if (!this._checkFrameLength()) return;

    this._stage  = frame.masked ? 3 : 4;
  },

  _checkFrameLength: function() {
    var length = this._message ? this._message.length : 0;

    if (length + this._frame.length > this._maxLength) {
      this._fail('too_large', 'WebSocket frame length too large');
      return false;
    } else {
      return true;
    }
  },

  _emitFrame: function(buffer) {
    var frame   = this._frame,
        payload = frame.payload = Hybi.mask(buffer, frame.maskingKey),
        opcode  = frame.opcode,
        message,
        code, reason,
        callbacks, callback;

    delete this._frame;

    if (opcode === this.OPCODES.continuation) {
      if (!this._message) return this._fail('protocol_error', 'Received unexpected continuation frame');
      this._message.pushFrame(frame);
    }

    if (opcode === this.OPCODES.text || opcode === this.OPCODES.binary) {
      this._message = new Message();
      this._message.pushFrame(frame);
    }

    if (frame.final && this.MESSAGE_OPCODES.indexOf(opcode) >= 0)
      return this._emitMessage(this._message);

    if (opcode === this.OPCODES.close) {
      code   = (payload.length >= 2) ? 256 * payload[0] + payload[1] : null;
      reason = (payload.length > 2) ? this._encode(payload.slice(2)) : null;

      if (!(payload.length === 0) &&
          !(code !== null && code >= this.MIN_RESERVED_ERROR && code <= this.MAX_RESERVED_ERROR) &&
          this.ERROR_CODES.indexOf(code) < 0)
        code = this.ERRORS.protocol_error;

      if (payload.length > 125 || (payload.length > 2 && !reason))
        code = this.ERRORS.protocol_error;

      this._shutdown(code, reason || '');
    }

    if (opcode === this.OPCODES.ping) {
      this.frame(payload, 'pong');
    }

    if (opcode === this.OPCODES.pong) {
      callbacks = this._pingCallbacks;
      message   = this._encode(payload);
      callback  = callbacks[message];

      delete callbacks[message];
      if (callback) callback()
    }
  },

  _emitMessage: function(message) {
    var message = this._message;
    message.read();

    delete this._message;

    this._extensions.processIncomingMessage(message, function(error, message) {
      if (error) return this._fail('extension_error', error.message);

      var payload = message.data;
      if (message.opcode === this.OPCODES.text) payload = this._encode(payload);

      if (payload === null)
        return this._fail('encoding_error', 'Could not decode a text frame as UTF-8');
      else
        this.emit('message', new Base.MessageEvent(payload));
    }, this);
  },

  _encode: function(buffer) {
    try {
      var string = buffer.toString('binary', 0, buffer.length);
      if (!this.UTF8_MATCH.test(string)) return null;
    } catch (e) {}
    return buffer.toString('utf8', 0, buffer.length);
  },

  _getInteger: function(bytes) {
    var number = 0;
    for (var i = 0, n = bytes.length; i < n; i++)
      number += bytes[i] * this.TWO_POWERS[n - 1 - i];
    return number;
  }
};

for (var key in instance)
  Hybi.prototype[key] = instance[key];

module.exports = Hybi;
