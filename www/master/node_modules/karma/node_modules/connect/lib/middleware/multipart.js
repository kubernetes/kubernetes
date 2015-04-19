/*!
 * Connect - multipart
 * Copyright(c) 2010 Sencha Inc.
 * Copyright(c) 2011 TJ Holowaychuk
 * MIT Licensed
 */

/**
 * Module dependencies.
 */

var formidable = require('formidable')
  , _limit = require('./limit')
  , utils = require('../utils')
  , qs = require('qs');

/**
 * noop middleware.
 */

function noop(req, res, next) {
  next();
}

/**
 * Multipart:
 * 
 * Parse multipart/form-data request bodies,
 * providing the parsed object as `req.body`
 * and `req.files`.
 *
 * Configuration:
 *
 *  The options passed are merged with [formidable](https://github.com/felixge/node-formidable)'s
 *  `IncomingForm` object, allowing you to configure the upload directory,
 *  size limits, etc. For example if you wish to change the upload dir do the following.
 *
 *     app.use(connect.multipart({ uploadDir: path }));
 *
 * Options:
 *
 *   - `limit`  byte limit defaulting to none
 *   - `defer`  defers processing and exposes the Formidable form object as `req.form`.
 *              `next()` is called without waiting for the form's "end" event.
 *              This option is useful if you need to bind to the "progress" event, for example.
 *
 * @param {Object} options
 * @return {Function}
 * @api public
 */

exports = module.exports = function(options){
  options = options || {};

  var limit = options.limit
    ? _limit(options.limit)
    : noop;

  return function multipart(req, res, next) {
    if (req._body) return next();
    req.body = req.body || {};
    req.files = req.files || {};

    if (!utils.hasBody(req)) return next();

    // ignore GET
    if ('GET' == req.method || 'HEAD' == req.method) return next();

    // check Content-Type
    if ('multipart/form-data' != utils.mime(req)) return next();

    // flag as parsed
    req._body = true;

    // parse
    limit(req, res, function(err){
      if (err) return next(err);

      var form = new formidable.IncomingForm
        , data = {}
        , files = {}
        , done;

      Object.keys(options).forEach(function(key){
        form[key] = options[key];
      });

      function ondata(name, val, data){
        if (Array.isArray(data[name])) {
          data[name].push(val);
        } else if (data[name]) {
          data[name] = [data[name], val];
        } else {
          data[name] = val;
        }
      }

      form.on('field', function(name, val){
        ondata(name, val, data);
      });

      form.on('file', function(name, val){
        ondata(name, val, files);
      });

      form.on('error', function(err){
        if (!options.defer) {
          err.status = 400;
          next(err);
        }
        done = true;
      });

      form.on('end', function(){
        if (done) return;
        try {
          req.body = qs.parse(data);
          req.files = qs.parse(files);
        } catch (err) {
          form.emit('error', err);
          return;
        }
        if (!options.defer) next();
      });

      form.parse(req);

      if (options.defer) {
        req.form = form;
        next();
      }
    });
  }
};
