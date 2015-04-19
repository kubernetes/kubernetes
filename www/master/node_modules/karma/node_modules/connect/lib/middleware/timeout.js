/*!
 * Connect - timeout
 * Ported from https://github.com/LearnBoost/connect-timeout
 * MIT Licensed
 */

/**
 * Module dependencies.
 */

var debug = require('debug')('connect:timeout');

/**
 * Timeout:
 *
 * Times out the request in `ms`, defaulting to `5000`. The
 * method `req.clearTimeout()` is added to revert this behaviour
 * programmatically within your application's middleware, routes, etc.
 *
 * The timeout error is passed to `next()` so that you may customize
 * the response behaviour. This error has the `.timeout` property as
 * well as `.status == 503`.
 *
 * @param {Number} ms
 * @return {Function}
 * @api public
 */

module.exports = function timeout(ms) {
  ms = ms || 5000;

  return function(req, res, next) {
    var id = setTimeout(function(){
      req.emit('timeout', ms);
    }, ms);

    req.on('timeout', function(){
      if (res.headerSent) return debug('response started, cannot timeout');
      var err = new Error('Response timeout');
      err.timeout = ms;
      err.status = 503;
      next(err);
    });

    req.clearTimeout = function(){
      clearTimeout(id);
    };

    res.on('header', function(){
      clearTimeout(id);
    });

    next();
  };
};
