'use strict';

var StreamReader = function() {
  this._queue     = [];
  this._queueSize = 0;
};

StreamReader.prototype.put = function(buffer) {
  if (!buffer || buffer.length === 0) return;
  if (!buffer.copy) buffer = new Buffer(buffer);
  this._queue.push(buffer);
  this._queueSize += buffer.length;
};

StreamReader.prototype.read = function(length) {
  if (length > this._queueSize) return null;
  if (length === 0) return new Buffer(0);
  
  var queue = this._queue,
      first = queue[0],
      buffer;
  
  if (first.length >= length) {
    this._queueSize -= length;
    if (first.length === length) {
      return queue.shift();
    } else {
      buffer = first.slice(0, length);
      queue[0] = first.slice(length);
      return buffer;
    }
  }
  
  var remain = length, buffers;

  for (var i=0, n = queue.length; i < n; i++) {
    if (remain < queue[i].length) break;
    remain -= queue[i].length;
  }
  buffers = queue.splice(0, i);

  if (remain > 0 && queue.length > 0) {
    buffers.push(queue[0].slice(0, remain));
    queue[0] = queue[0].slice(remain);
  }
  this._queueSize -= length;
  return this._concat(buffers, length);
};

StreamReader.prototype._concat = function(buffers, length) {
  if (Buffer.concat) return Buffer.concat(buffers, length);

  var buffer = new Buffer(length),
      offset = 0;

  for (var i = 0, n = buffers.length; i < n; i++) {
    buffers[i].copy(buffer, offset);
    offset += buffers[i].length;
  }
  return buffer;
};

module.exports = StreamReader;
