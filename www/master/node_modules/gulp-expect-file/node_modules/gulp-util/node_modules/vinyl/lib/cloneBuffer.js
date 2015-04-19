var Buffer = require('buffer').Buffer;

module.exports = function(buf) {
  var out = new Buffer(buf.length);
  buf.copy(out);
  return out;
};