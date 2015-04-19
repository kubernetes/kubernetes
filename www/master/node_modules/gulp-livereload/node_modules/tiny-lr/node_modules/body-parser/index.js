/*!
 * body-parser
 * Copyright(c) 2014 Douglas Christopher Wilson
 * MIT Licensed
 */

/**
 * Module dependencies.
 */

var deprecate = require('depd')('body-parser')
var fs = require('fs')
var path = require('path')

/**
 * Module exports.
 */

exports = module.exports = deprecate.function(bodyParser,
  'bodyParser: use individual json/urlencoded middlewares')

/**
 * Path to the parser modules.
 */

var parsersDir = path.join(__dirname, 'lib', 'types')

/**
 * Auto-load bundled parsers with getters.
 */

fs.readdirSync(parsersDir).forEach(function onfilename(filename) {
  if (!/\.js$/.test(filename)) return

  var loc = path.resolve(parsersDir, filename)
  var mod
  var name = path.basename(filename, '.js')

  function load() {
    if (mod) {
      return mod
    }

    return mod = require(loc)
  }

  Object.defineProperty(exports, name, {
    configurable: true,
    enumerable: true,
    get: load
  })
})

/**
 * Create a middleware to parse json and urlencoded bodies.
 *
 * @param {object} [options]
 * @return {function}
 * @deprecated
 * @api public
 */

function bodyParser(options){
  var opts = {}

  options = options || {}

  // exclude type option
  for (var prop in options) {
    if ('type' !== prop) {
      opts[prop] = options[prop]
    }
  }

  var _urlencoded = exports.urlencoded(opts)
  var _json = exports.json(opts)

  return function bodyParser(req, res, next) {
    _json(req, res, function(err){
      if (err) return next(err);
      _urlencoded(req, res, next);
    });
  }
}
