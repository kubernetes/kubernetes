'use strict';

var Base = require('./base'),
    util = require('util');

var Draft75 = function(request, url, options) {
  Base.apply(this, arguments);
  this._stage = 0;
  this.version = 'hixie-75';

  this._headers.set('Upgrade', 'WebSocket');
  this._headers.set('Connection', 'Upgrade');
  this._headers.set('WebSocket-Origin', this._request.headers.origin);
  this._headers.set('WebSocket-Location', this.url);
};
util.inherits(Draft75, Base);

var instance = {
  close: function() {
    if (this.readyState === 3) return false;
    this.readyState = 3;
    this.emit('close', new Base.CloseEvent(null, null));
    return true;
  },

  parse: function(buffer) {
    if (this.readyState > 1) return;

    var data, message, value;
    for (var i = 0, n = buffer.length; i < n; i++) {
      data = buffer[i];

      switch (this._stage) {
        case -1:
          this._body.push(data);
          this._sendHandshakeBody();
          break;

        case 0:
          this._parseLeadingByte(data);
          break;

        case 1:
          value = (data & 0x7F);
          this._length = value + 128 * this._length;

          if (this._closing && this._length === 0) {
            return this.close();
          }
          else if ((0x80 & data) !== 0x80) {
            if (this._length === 0) {
              this._stage = 0;
            }
            else {
              this._skipped = 0;
              this._stage = 2;
            }
          }
          break;

        case 2:
          if (data === 0xFF) {
            message = new Buffer(this._buffer).toString('utf8', 0, this._buffer.length);
            this.emit('message', new Base.MessageEvent(message));
            this._stage = 0;
          }
          else {
            if (this._length) {
              this._skipped += 1;
              if (this._skipped === this._length)
                this._stage = 0;
            } else {
              this._buffer.push(data);
              if (this._buffer.length > this._maxLength) return this.close();
            }
          }
          break;
      }
    }
  },

  frame: function(data) {
    if (this.readyState === 0) return this._queue([data]);
    if (this.readyState > 1) return false;

    var buffer = new Buffer(data, 'utf8'),
        frame  = new Buffer(buffer.length + 2);

    frame[0] = 0x00;
    frame[buffer.length + 1] = 0xFF;
    buffer.copy(frame, 1);

    this._write(frame);
    return true;
  },

  _handshakeResponse: function() {
    var start   = 'HTTP/1.1 101 Web Socket Protocol Handshake',
        headers = [start, this._headers.toString(), ''];

    return new Buffer(headers.join('\r\n'), 'utf8');
  },

  _parseLeadingByte: function(data) {
    if ((0x80 & data) === 0x80) {
      this._length = 0;
      this._stage = 1;
    } else {
      delete this._length;
      delete this._skipped;
      this._buffer = [];
      this._stage = 2;
    }
  }
};

for (var key in instance)
  Draft75.prototype[key] = instance[key];

module.exports = Draft75;
