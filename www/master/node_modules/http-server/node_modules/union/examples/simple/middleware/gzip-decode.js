var spawn = require('child_process').spawn,
    util = require('util'),
    RequestStream = require('../../lib').RequestStream;

var GzipDecode = module.exports = function GzipDecoder(options) {
  RequestStream.call(this, options);

  this.on('pipe', this.decode);
}

util.inherits(GzipDecode, RequestStream);

GzipDecode.prototype.decode = function (source) {
  this.decoder = spawn('gunzip');
  this.decoder.stdout.on('data', this._onGunzipData.bind(this));
  this.decoder.stdout.on('end', this._onGunzipEnd.bind(this));
  source.pipe(this.decoder);
}

GzipDecoderStack.prototype._onGunzipData = function (chunk) {
  this.emit('data', chunk);
}

GzipDecoderStack.prototype._onGunzipEnd = function () {
  this.emit('end');
}