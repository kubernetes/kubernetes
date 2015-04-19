
/*!
 * Connect - json
 * Copyright(c) 2010 Sencha Inc.
 * Copyright(c) 2011 TJ Holowaychuk
 * MIT Licensed
 */

/**
 * Module dependencies.
 */

var utils = require('../utils')
  , _limit = require('./limit');

/**
 * noop middleware.
 */

function noop(req, res, next) {
  next();
}

/**
 * JSON:
 *
 * Parse JSON request bodies, providing the
 * parsed object as `req.body`.
 *
 * Options:
 *
 *   - `strict`  when `false` anything `JSON.parse()` accepts will be parsed
 *   - `reviver`  used as the second "reviver" argument for JSON.parse
 *   - `limit`  byte limit disabled by default
 *
 * @param {Object} options
 * @return {Function}
 * @api public
 */

exports = module.exports = function(options){
  var options = options || {}
    , strict = options.strict !== false;

  var limit = options.limit
    ? _limit(options.limit)
    : noop;

  return function json(req, res, next) {
    if (req._body) return next();
    req.body = req.body || {};

    if (!utils.hasBody(req)) return next();

    // check Content-Type
    if (!exports.regexp.test(utils.mime(req))) return next();

    // flag as parsed
    req._body = true;

    // parse
    limit(req, res, function(err){
      if (err) return next(err);
      var buf = '';
      req.setEncoding('utf8');
      req.on('data', function(chunk){ buf += chunk });
      req.on('end', function(){
        var first = buf.trim()[0];

        if (0 == buf.length) {
          return next(utils.error(400, 'invalid json, empty body'));
        }
        
        if (strict && '{' != first && '[' != first) return next(utils.error(400, 'invalid json'));
        try {
          req.body = JSON.parse(buf, options.reviver);
        } catch (err){
          err.body = buf;
          err.status = 400;
          return next(err);
        }
        next();
      });
    });
  };
};

exports.regexp = /^application\/([\w!#\$%&\*`\-\.\^~]*\+)?json$/i;

