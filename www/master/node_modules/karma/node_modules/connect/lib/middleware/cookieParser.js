
/*!
 * Connect - cookieParser
 * Copyright(c) 2010 Sencha Inc.
 * Copyright(c) 2011 TJ Holowaychuk
 * MIT Licensed
 */

/**
 * Module dependencies.
 */

var utils = require('./../utils')
  , cookie = require('cookie');

/**
 * Cookie parser:
 *
 * Parse _Cookie_ header and populate `req.cookies`
 * with an object keyed by the cookie names. Optionally
 * you may enabled signed cookie support by passing
 * a `secret` string, which assigns `req.secret` so
 * it may be used by other middleware.
 *
 * Examples:
 *
 *     connect()
 *       .use(connect.cookieParser('optional secret string'))
 *       .use(function(req, res, next){
 *         res.end(JSON.stringify(req.cookies));
 *       })
 *
 * @param {String} secret
 * @return {Function}
 * @api public
 */

module.exports = function cookieParser(secret){
  return function cookieParser(req, res, next) {
    if (req.cookies) return next();
    var cookies = req.headers.cookie;

    req.secret = secret;
    req.cookies = {};
    req.signedCookies = {};

    if (cookies) {
      try {
        req.cookies = cookie.parse(cookies);
        if (secret) {
          req.signedCookies = utils.parseSignedCookies(req.cookies, secret);
          req.signedCookies = utils.parseJSONCookies(req.signedCookies);
        }
        req.cookies = utils.parseJSONCookies(req.cookies);
      } catch (err) {
        err.status = 400;
        return next(err);
      }
    }
    next();
  };
};
