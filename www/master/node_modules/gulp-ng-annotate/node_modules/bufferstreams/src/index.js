var PassThrough = require('readable-stream').PassThrough;
var util = require('util');

// Inherit of PassThrough stream
util.inherits(BufferStream, PassThrough);

// Constructor
function BufferStream(cb) {

  // Ensure new were used
  if (!(this instanceof BufferStream)) {
    return new BufferStream(cb);
  }

  // Parent constructor
  PassThrough.call(this);

  // Keep a reference to the callback
  this._cb = cb;

  // Internal buffer
  this._buf = Buffer('');
}

BufferStream.prototype._transform = function(chunk, encoding, done) {

  this._buf = Buffer.concat([this._buf, chunk], this._buf.length + chunk.length);

  done();

};

BufferStream.prototype._flush = function(done) {
  var _that = this;

  this._cb(null, this._buf, function(err, buf) {
    if (buf && buf.length) {
      _that.push(buf);
    }
    done();
  });

};

module.exports = BufferStream;
