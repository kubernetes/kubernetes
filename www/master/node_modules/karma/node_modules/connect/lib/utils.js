
/*!
 * Connect - utils
 * Copyright(c) 2010 Sencha Inc.
 * Copyright(c) 2011 TJ Holowaychuk
 * MIT Licensed
 */

/**
 * Module dependencies.
 */

var http = require('http')
  , crypto = require('crypto')
  , parse = require('url').parse
  , signature = require('cookie-signature')
  , nodeVersion = process.versions.node.split('.');

// pause is broken in node < 0.10
exports.brokenPause = parseInt(nodeVersion[0], 10) === 0
  && parseInt(nodeVersion[1], 10) < 10;

/**
 * Return `true` if the request has a body, otherwise return `false`.
 *
 * @param  {IncomingMessage} req
 * @return {Boolean}
 * @api private
 */

exports.hasBody = function(req) {
  var encoding = 'transfer-encoding' in req.headers;
  var length = 'content-length' in req.headers && req.headers['content-length'] !== '0';
  return encoding || length;
};

/**
 * Extract the mime type from the given request's
 * _Content-Type_ header.
 *
 * @param  {IncomingMessage} req
 * @return {String}
 * @api private
 */

exports.mime = function(req) {
  var str = req.headers['content-type'] || '';
  return str.split(';')[0];
};

/**
 * Generate an `Error` from the given status `code`
 * and optional `msg`.
 *
 * @param {Number} code
 * @param {String} msg
 * @return {Error}
 * @api private
 */

exports.error = function(code, msg){
  var err = new Error(msg || http.STATUS_CODES[code]);
  err.status = code;
  return err;
};

/**
 * Return md5 hash of the given string and optional encoding,
 * defaulting to hex.
 *
 *     utils.md5('wahoo');
 *     // => "e493298061761236c96b02ea6aa8a2ad"
 *
 * @param {String} str
 * @param {String} encoding
 * @return {String}
 * @api private
 */

exports.md5 = function(str, encoding){
  return crypto
    .createHash('md5')
    .update(str)
    .digest(encoding || 'hex');
};

/**
 * Merge object b with object a.
 *
 *     var a = { foo: 'bar' }
 *       , b = { bar: 'baz' };
 *
 *     utils.merge(a, b);
 *     // => { foo: 'bar', bar: 'baz' }
 *
 * @param {Object} a
 * @param {Object} b
 * @return {Object}
 * @api private
 */

exports.merge = function(a, b){
  if (a && b) {
    for (var key in b) {
      a[key] = b[key];
    }
  }
  return a;
};

/**
 * Escape the given string of `html`.
 *
 * @param {String} html
 * @return {String}
 * @api private
 */

exports.escape = function(html){
  return String(html)
    .replace(/&(?!\w+;)/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
};

/**
 * Sign the given `val` with `secret`.
 *
 * @param {String} val
 * @param {String} secret
 * @return {String}
 * @api private
 */

exports.sign = function(val, secret){
  console.warn('do not use utils.sign(), use https://github.com/visionmedia/node-cookie-signature')
  return val + '.' + crypto
    .createHmac('sha256', secret)
    .update(val)
    .digest('base64')
    .replace(/=+$/, '');
};

/**
 * Unsign and decode the given `val` with `secret`,
 * returning `false` if the signature is invalid.
 *
 * @param {String} val
 * @param {String} secret
 * @return {String|Boolean}
 * @api private
 */

exports.unsign = function(val, secret){
  console.warn('do not use utils.unsign(), use https://github.com/visionmedia/node-cookie-signature')
  var str = val.slice(0, val.lastIndexOf('.'));
  return exports.sign(str, secret) == val
    ? str
    : false;
};

/**
 * Parse signed cookies, returning an object
 * containing the decoded key/value pairs,
 * while removing the signed key from `obj`.
 *
 * @param {Object} obj
 * @return {Object}
 * @api private
 */

exports.parseSignedCookies = function(obj, secret){
  var ret = {};
  Object.keys(obj).forEach(function(key){
    var val = obj[key];
    if (0 == val.indexOf('s:')) {
      val = signature.unsign(val.slice(2), secret);
      if (val) {
        ret[key] = val;
        delete obj[key];
      }
    }
  });
  return ret;
};

/**
 * Parse a signed cookie string, return the decoded value
 *
 * @param {String} str signed cookie string
 * @param {String} secret
 * @return {String} decoded value
 * @api private
 */

exports.parseSignedCookie = function(str, secret){
  return 0 == str.indexOf('s:')
    ? signature.unsign(str.slice(2), secret)
    : str;
};

/**
 * Parse JSON cookies.
 *
 * @param {Object} obj
 * @return {Object}
 * @api private
 */

exports.parseJSONCookies = function(obj){
  Object.keys(obj).forEach(function(key){
    var val = obj[key];
    var res = exports.parseJSONCookie(val);
    if (res) obj[key] = res;
  });
  return obj;
};

/**
 * Parse JSON cookie string
 *
 * @param {String} str
 * @return {Object} Parsed object or null if not json cookie
 * @api private
 */

exports.parseJSONCookie = function(str) {
  if (0 == str.indexOf('j:')) {
    try {
      return JSON.parse(str.slice(2));
    } catch (err) {
      // no op
    }
  }
};

/**
 * Pause `data` and `end` events on the given `obj`.
 * Middleware performing async tasks _should_ utilize
 * this utility (or similar), to re-emit data once
 * the async operation has completed, otherwise these
 * events may be lost. Pause is only required for
 * node versions less than 10, and is replaced with
 * noop's otherwise.
 *
 *      var pause = utils.pause(req);
 *      fs.readFile(path, function(){
 *         next();
 *         pause.resume();
 *      });
 *
 * @param {Object} obj
 * @return {Object}
 * @api private
 */

exports.pause = exports.brokenPause
  ? require('pause')
  : function () {
    return {
      end: noop,
      resume: noop
    }
  }

/**
 * Strip `Content-*` headers from `res`.
 *
 * @param {ServerResponse} res
 * @api private
 */

exports.removeContentHeaders = function(res){
  Object.keys(res._headers).forEach(function(field){
    if (0 == field.indexOf('content')) {
      res.removeHeader(field);
    }
  });
};

/**
 * Check if `req` is a conditional GET request.
 *
 * @param {IncomingMessage} req
 * @return {Boolean}
 * @api private
 */

exports.conditionalGET = function(req) {
  return req.headers['if-modified-since']
    || req.headers['if-none-match'];
};

/**
 * Respond with 401 "Unauthorized".
 *
 * @param {ServerResponse} res
 * @param {String} realm
 * @api private
 */

exports.unauthorized = function(res, realm) {
  res.statusCode = 401;
  res.setHeader('WWW-Authenticate', 'Basic realm="' + realm + '"');
  res.end('Unauthorized');
};

/**
 * Respond with 304 "Not Modified".
 *
 * @param {ServerResponse} res
 * @param {Object} headers
 * @api private
 */

exports.notModified = function(res) {
  exports.removeContentHeaders(res);
  res.statusCode = 304;
  res.end();
};

/**
 * Return an ETag in the form of `"<size>-<mtime>"`
 * from the given `stat`.
 *
 * @param {Object} stat
 * @return {String}
 * @api private
 */

exports.etag = function(stat) {
  return '"' + stat.size + '-' + Number(stat.mtime) + '"';
};

/**
 * Parse the given Cache-Control `str`.
 *
 * @param {String} str
 * @return {Object}
 * @api private
 */

exports.parseCacheControl = function(str){
  var directives = str.split(',')
    , obj = {};

  for(var i = 0, len = directives.length; i < len; i++) {
    var parts = directives[i].split('=')
      , key = parts.shift().trim()
      , val = parseInt(parts.shift(), 10);

    obj[key] = isNaN(val) ? true : val;
  }

  return obj;
};

/**
 * Parse the `req` url with memoization.
 *
 * @param {ServerRequest} req
 * @return {Object}
 * @api private
 */

exports.parseUrl = function(req){
  var parsed = req._parsedUrl;
  if (parsed && parsed.href == req.url) {
    return parsed;
  } else {
    return req._parsedUrl = parse(req.url);
  }
};

/**
 * Parse byte `size` string.
 *
 * @param {String} size
 * @return {Number}
 * @api private
 */

exports.parseBytes = require('bytes');

function noop() {}
