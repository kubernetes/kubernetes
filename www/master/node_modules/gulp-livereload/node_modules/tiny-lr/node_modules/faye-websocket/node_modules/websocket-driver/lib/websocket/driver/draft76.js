'use strict';

var Base    = require('./base'),
    Draft75 = require('./draft75'),
    crypto  = require('crypto'),
    util    = require('util');


var numberFromKey = function(key) {
  return parseInt(key.match(/[0-9]/g).join(''), 10);
};

var spacesInKey = function(key) {
  return key.match(/ /g).length;
};

var bigEndian = function(number) {
  var string = '';
  [24, 16, 8, 0].forEach(function(offset) {
    string += String.fromCharCode(number >> offset & 0xFF);
  });
  return string;
};


var Draft76 = function(request, url, options) {
  Draft75.apply(this, arguments);
  this._stage  = -1;
  this._body   = [];
  this.version = 'hixie-76';

  this._headers.clear();

  this._headers.set('Upgrade', 'WebSocket');
  this._headers.set('Connection', 'Upgrade');
  this._headers.set('Sec-WebSocket-Origin', this._request.headers.origin);
  this._headers.set('Sec-WebSocket-Location', this.url);
};
util.inherits(Draft76, Draft75);

var instance = {
  BODY_SIZE: 8,

  start: function() {
    if (!Draft75.prototype.start.call(this)) return false;
    this._started = true;
    this._sendHandshakeBody();
    return true;
  },

  close: function() {
    if (this.readyState === 3) return false;
    this._write(new Buffer([0xFF, 0x00]));
    this.readyState = 3;
    this.emit('close', new Base.CloseEvent(null, null));
    return true;
  },

  _handshakeResponse: function() {
    var start   = 'HTTP/1.1 101 WebSocket Protocol Handshake',
        headers = [start, this._headers.toString(), ''];

    return new Buffer(headers.join('\r\n'), 'binary');
  },

  _handshakeSignature: function() {
    if (this._body.length < this.BODY_SIZE) return null;
    var body = new Buffer(this._body.slice(0, this.BODY_SIZE));

    var headers = this._request.headers,
        key1    = headers['sec-websocket-key1'],
        value1  = numberFromKey(key1) / spacesInKey(key1),
        key2    = headers['sec-websocket-key2'],
        value2  = numberFromKey(key2) / spacesInKey(key2),
        md5     = crypto.createHash('md5');

    md5.update(bigEndian(value1));
    md5.update(bigEndian(value2));
    md5.update(body.toString('binary'));

    return new Buffer(md5.digest('binary'), 'binary');
  },

  _sendHandshakeBody: function() {
    if (!this._started) return;
    var signature = this._handshakeSignature();
    if (!signature) return;

    this._write(signature);
    this._stage = 0;
    this._open();

    if (this._body.length > this.BODY_SIZE)
      this.parse(this._body.slice(this.BODY_SIZE));
  },

  _parseLeadingByte: function(data) {
    if (data !== 0xFF)
      return Draft75.prototype._parseLeadingByte.call(this, data);

    this._closing = true;
    this._length  = 0;
    this._stage   = 1;
  }
};

for (var key in instance)
  Draft76.prototype[key] = instance[key];

module.exports = Draft76;
