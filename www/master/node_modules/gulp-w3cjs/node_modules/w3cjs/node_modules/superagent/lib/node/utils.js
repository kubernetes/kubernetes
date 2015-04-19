
/**
 * Module dependencies.
 */

var StringDecoder = require('string_decoder').StringDecoder;
var Stream = require('stream');
var zlib;

/**
 * Require zlib module for Node 0.6+
 */

try {
  zlib = require('zlib');
} catch (e) { }

/**
 * Generate a UID with the given `len`.
 *
 * @param {Number} len
 * @return {String}
 * @api private
 */

exports.uid = function(len){
  var buf = '';
  var chars = 'abcdefghijklmnopqrstuvwxyz123456789';
  var nchars = chars.length;
  while (len--) buf += chars[Math.random() * nchars | 0];
  return buf;
};

/**
 * Return the mime type for the given `str`.
 *
 * @param {String} str
 * @return {String}
 * @api private
 */

exports.type = function(str){
  return str.split(/ *; */).shift();
};

/**
 * Return header field parameters.
 *
 * @param {String} str
 * @return {Object}
 * @api private
 */

exports.params = function(str){
  return str.split(/ *; */).reduce(function(obj, str){
    var parts = str.split(/ *= */);
    var key = parts.shift();
    var val = parts.shift();

    if (key && val) obj[key] = val;
    return obj;
  }, {});
};

/**
 * Parse Link header fields.
 *
 * @param {String} str
 * @return {Object}
 * @api private
 */

exports.parseLinks = function(str){
  return str.split(/ *, */).reduce(function(obj, str){
    var parts = str.split(/ *; */);
    var url = parts[0].slice(1, -1);
    var rel = parts[1].split(/ *= */)[1].slice(1, -1);
    obj[rel] = url;
    return obj;
  }, {});
};

/**
 * Buffers response data events and re-emits when they're unzipped.
 *
 * @param {Request} req
 * @param {Response} res
 * @api private
 */

exports.unzip = function(req, res){
  if (!zlib) return;

  var unzip = zlib.createUnzip();
  var stream = new Stream;
  var decoder;

  // make node responseOnEnd() happy
  stream.req = req;

  unzip.on('error', function(err){
    stream.emit('error', err);
  });

  // pipe to unzip
  res.pipe(unzip);

  // override `setEncoding` to capture encoding
  res.setEncoding = function(type){
    decoder = new StringDecoder(type);
  };

  // decode upon decompressing with captured encoding
  unzip.on('data', function(buf){
    if (decoder) {
      var str = decoder.write(buf);
      if (str.length) stream.emit('data', str);
    } else {
      stream.emit('data', buf);
    }
  });

  unzip.on('end', function(){
    stream.emit('end');
  });

  // override `on` to capture data listeners
  var _on = res.on;
  res.on = function(type, fn){
    if ('data' == type || 'end' == type) {
      stream.on(type, fn);
    } else if ('error' == type) {
      stream.on(type, fn);
      _on.call(res, type, fn);
    } else {
      _on.call(res, type, fn);
    }
  };
};

/**
 * Strip content related fields from `header`.
 *
 * @param {Object} header
 * @return {Object} header
 * @api private
 */

exports.cleanHeader = function(header){
  delete header['content-type'];
  delete header['content-length'];
  delete header['transfer-encoding'];
  delete header['cookie'];
  delete header['host'];
  return header;
};
