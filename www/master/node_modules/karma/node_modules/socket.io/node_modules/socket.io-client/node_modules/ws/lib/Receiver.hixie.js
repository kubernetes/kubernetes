/*!
 * ws: a node.js websocket client
 * Copyright(c) 2011 Einar Otto Stangvik <einaros@gmail.com>
 * MIT Licensed
 */

var util = require('util');

/**
 * State constants
 */

var EMPTY = 0
  , BODY = 1;
var BINARYLENGTH = 2
  , BINARYBODY = 3;

/**
 * Hixie Receiver implementation
 */

function Receiver () {
  this.state = EMPTY;
  this.buffers = [];
  this.messageEnd = -1;
  this.spanLength = 0;
  this.dead = false;

  this.onerror = function() {};
  this.ontext = function() {};
  this.onbinary = function() {};
  this.onclose = function() {};
  this.onping = function() {};
  this.onpong = function() {};
}

module.exports = Receiver;

/**
 * Add new data to the parser.
 *
 * @api public
 */

Receiver.prototype.add = function(data) {
  var self = this;
  function doAdd() {
    if (self.state === EMPTY) {
      if (data.length == 2 && data[0] == 0xFF && data[1] == 0x00) {
        self.reset();
        self.onclose();
        return;
      }
      if (data[0] === 0x80) {
        self.messageEnd = 0;
        self.state = BINARYLENGTH;
        data = data.slice(1);
      } else {

      if (data[0] !== 0x00) {
        self.error('payload must start with 0x00 byte', true);
        return;
      }
      data = data.slice(1);
      self.state = BODY;

      }
    }
    if (self.state === BINARYLENGTH) {
      var i = 0;
      while ((i < data.length) && (data[i] & 0x80)) {
        self.messageEnd = 128 * self.messageEnd + (data[i] & 0x7f);
        ++i;
      }
      if (i < data.length) {
        self.messageEnd = 128 * self.messageEnd + (data[i] & 0x7f);
        self.state = BINARYBODY;
        ++i;
      }
      if (i > 0)
        data = data.slice(i);
    }
    if (self.state === BINARYBODY) {
      var dataleft = self.messageEnd - self.spanLength;
      if (data.length >= dataleft) {
        // consume the whole buffer to finish the frame
        self.buffers.push(data);
        self.spanLength += dataleft;
        self.messageEnd = dataleft;
        return self.parse();
      }
      // frame's not done even if we consume it all
      self.buffers.push(data);
      self.spanLength += data.length;
      return;
    }
    self.buffers.push(data);
    if ((self.messageEnd = bufferIndex(data, 0xFF)) != -1) {
      self.spanLength += self.messageEnd;
      return self.parse();
    }
    else self.spanLength += data.length;
  }
  while(data) data = doAdd();
};

/**
 * Releases all resources used by the receiver.
 *
 * @api public
 */

Receiver.prototype.cleanup = function() {
  this.dead = true;
  this.state = EMPTY;
  this.buffers = [];
};

/**
 * Process buffered data.
 *
 * @api public
 */

Receiver.prototype.parse = function() {
  var output = new Buffer(this.spanLength);
  var outputIndex = 0;
  for (var bi = 0, bl = this.buffers.length; bi < bl - 1; ++bi) {
    var buffer = this.buffers[bi];
    buffer.copy(output, outputIndex);
    outputIndex += buffer.length;
  }
  var lastBuffer = this.buffers[this.buffers.length - 1];
  if (this.messageEnd > 0) lastBuffer.copy(output, outputIndex, 0, this.messageEnd);
  if (this.state !== BODY) --this.messageEnd;
  var tail = null;
  if (this.messageEnd < lastBuffer.length - 1) {
    tail = lastBuffer.slice(this.messageEnd + 1);
  }
  this.reset();
  this.ontext(output.toString('utf8'));
  return tail;
};

/**
 * Handles an error
 *
 * @api private
 */

Receiver.prototype.error = function (reason, terminate) {
  this.reset();
  this.onerror(reason, terminate);
  return this;
};

/**
 * Reset parser state
 *
 * @api private
 */

Receiver.prototype.reset = function (reason) {
  if (this.dead) return;
  this.state = EMPTY;
  this.buffers = [];
  this.messageEnd = -1;
  this.spanLength = 0;
};

/**
 * Internal api
 */

function bufferIndex(buffer, byte) {
  for (var i = 0, l = buffer.length; i < l; ++i) {
    if (buffer[i] === byte) return i;
  }
  return -1;
}
