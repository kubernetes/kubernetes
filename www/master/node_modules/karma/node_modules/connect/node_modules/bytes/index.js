
/**
 * Parse byte `size` string.
 *
 * @param {String} size
 * @return {Number}
 * @api public
 */

module.exports = function(size) {
  if ('number' == typeof size) return convert(size);
  var parts = size.match(/^(\d+(?:\.\d+)?) *(kb|mb|gb)$/)
    , n = parseFloat(parts[1])
    , type = parts[2];

  var map = {
      kb: 1 << 10
    , mb: 1 << 20
    , gb: 1 << 30
  };

  return map[type] * n;
};

/**
 * convert bytes into string.
 *
 * @param {Number} b - bytes to convert
 * @return {String}
 * @api public
 */

function convert (b) {
  var gb = 1 << 30, mb = 1 << 20, kb = 1 << 10;
  if (b >= gb) return (Math.round(b / gb * 100) / 100) + 'gb';
  if (b >= mb) return (Math.round(b / mb * 100) / 100) + 'mb';
  if (b >= kb) return (Math.round(b / kb * 100) / 100) + 'kb';
  return b + 'b';
}
