/*!
 * Connect - csrf
 * Copyright(c) 2011 Sencha Inc.
 * MIT Licensed
 */

/**
 * Module dependencies.
 */

var utils = require('../utils');
var uid = require('uid2');
var crypto = require('crypto');

/**
 * Anti CSRF:
 *
 * CSRF protection middleware.
 *
 * This middleware adds a `req.csrfToken()` function to make a token
 * which should be added to requests which mutate
 * state, within a hidden form field, query-string etc. This
 * token is validated against the visitor's session.
 *
 * The default `value` function checks `req.body` generated
 * by the `bodyParser()` middleware, `req.query` generated
 * by `query()`, and the "X-CSRF-Token" header field.
 *
 * This middleware requires session support, thus should be added
 * somewhere _below_ `session()` and `cookieParser()`.
 *
 * Options:
 *
 *    - `value` a function accepting the request, returning the token
 *
 * @param {Object} options
 * @api public
 */

module.exports = function csrf(options) {
  options = options || {};
  var value = options.value || defaultValue;

  return function(req, res, next){
    
    // already have one
    var secret = req.session._csrfSecret;
    if (secret) return createToken(secret);

    // generate secret
    uid(24, function(err, secret){
      if (err) return next(err);
      req.session._csrfSecret = secret;
      createToken(secret);
    });
    
    // generate the token
    function createToken(secret) {
      var token;

      // lazy-load token
      req.csrfToken = function csrfToken() {
        return token || (token = saltedToken(secret));
      };
      
      // compatibility with old middleware
      Object.defineProperty(req.session, '_csrf', {
        configurable: true,
        get: function() {
          console.warn('req.session._csrf is deprecated, use req.csrfToken([callback]) instead');
          return req.csrfToken();
        }
      });
      
      // ignore these methods
      if ('GET' == req.method || 'HEAD' == req.method || 'OPTIONS' == req.method) return next();
      
      // determine user-submitted value
      var val = value(req);
      
      // check
      if (!checkToken(val, secret)) return next(utils.error(403));
      
      next();
    }
  }
};

/**
 * Default value function, checking the `req.body`
 * and `req.query` for the CSRF token.
 *
 * @param {IncomingMessage} req
 * @return {String}
 * @api private
 */

function defaultValue(req) {
  return (req.body && req.body._csrf)
    || (req.query && req.query._csrf)
    || (req.headers['x-csrf-token'])
    || (req.headers['x-xsrf-token']);
}

/**
 * Return salted token.
 *
 * @param {String} secret
 * @return {String}
 * @api private
 */

function saltedToken(secret) {
  return createToken(generateSalt(10), secret);
}

/**
 * Creates a CSRF token from a given salt and secret.
 *
 * @param {String} salt (should be 10 characters)
 * @param {String} secret
 * @return {String}
 * @api private
 */

function createToken(salt, secret) {
  return salt + crypto
    .createHash('sha1')
    .update(salt + secret)
    .digest('base64');
}

/**
 * Checks if a given CSRF token matches the given secret.
 *
 * @param {String} token
 * @param {String} secret
 * @return {Boolean}
 * @api private
 */

function checkToken(token, secret) {
  if ('string' != typeof token) return false;
  return token === createToken(token.slice(0, 10), secret);
}

/**
 * Generates a random salt, using a fast non-blocking PRNG (Math.random()).
 *
 * @param {Number} length
 * @return {String}
 * @api private
 */

function generateSalt(length) {
  var i, r = [];
  for (i = 0; i < length; ++i) {
    r.push(SALTCHARS[Math.floor(Math.random() * SALTCHARS.length)]);
  }
  return r.join('');
}

var SALTCHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
