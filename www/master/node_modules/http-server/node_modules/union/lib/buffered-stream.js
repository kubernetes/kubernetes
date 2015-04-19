/*
 * buffered-stream.js: A simple(r) Stream which is partially buffered into memory.
 *
 * (C) 2010, Mikeal Rogers
 *
 * Adapted for Flatiron
 * (C) 2011, Nodejitsu Inc.
 * MIT LICENSE
 *
 */

var events = require('events'),
    fs = require('fs'),
    stream = require('stream'),
    util = require('util');

//
// ### function BufferedStream (limit)
// #### @limit {number} **Optional** Size of the buffer to limit
// Constructor function for the BufferedStream object responsible for
// maintaining a stream interface which can also persist to memory
// temporarily.
//

var BufferedStream = module.exports = function (limit) {
  events.EventEmitter.call(this);

  if (typeof limit === 'undefined') {
    limit = Infinity;
  }

  this.limit = limit;
  this.size = 0;
  this.chunks = [];
  this.writable = true;
  this.readable = true;
  this._buffer = true;
};

util.inherits(BufferedStream, stream.Stream);

Object.defineProperty(BufferedStream.prototype, 'buffer', {
  get: function () {
    return this._buffer;
  },
  set: function (value) {
    if (!value && this.chunks) {
      var self = this;
      this.chunks.forEach(function (c) { self.emit('data', c) });
      if (this.ended) this.emit('end');
      this.size = 0;
      delete this.chunks;
    }

    this._buffer = value;
  }
});

BufferedStream.prototype.pipe = function () {
  var self = this,
      dest;

  if (self.resume) {
    self.resume();
  }

  dest = stream.Stream.prototype.pipe.apply(self, arguments);

  //
  // just incase you are piping to two streams, do not emit data twice.
  // note: you can pipe twice, but you need to pipe both streams in the same tick.
  // (this is normal for streams)
  //
  if (this.piped) {
    return dest;
  }

  process.nextTick(function () {
    if (self.chunks) {
      self.chunks.forEach(function (c) { self.emit('data', c) });
      self.size = 0;
      delete self.chunks;
    }

    if (!self.readable) {
      if (self.ended) {
        self.emit('end');
      }
      else if (self.closed) {
        self.emit('close');
      }
    }
  });

  this.piped = true;

  return dest;
};

BufferedStream.prototype.write = function (chunk) {
  if (!this.chunks || this.piped) {
    this.emit('data', chunk);
    return;
  }

  this.chunks.push(chunk);
  this.size += chunk.length;
  if (this.limit < this.size) {
    this.pause();
  }
};

BufferedStream.prototype.end = function () {
  this.readable = false;
  this.ended = true;
  this.emit('end');
};

BufferedStream.prototype.destroy = function () {
  this.readable = false;
  this.writable = false;
  delete this.chunks;
};

BufferedStream.prototype.close = function () {
  this.readable = false;
  this.closed = true;
};

if (!stream.Stream.prototype.pause) {
  BufferedStream.prototype.pause = function () {
    this.emit('pause');
  };
}

if (!stream.Stream.prototype.resume) {
  BufferedStream.prototype.resume = function () {
    this.emit('resume');
  };
}

