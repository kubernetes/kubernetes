/*!
 * body-parser
 * Copyright(c) 2014 Douglas Christopher Wilson
 * MIT Licensed
 */

/**
 * Module dependencies.
 */

var bytes = require('bytes')
var read = require('../read')
var typeis = require('type-is')

/**
 * Module exports.
 */

module.exports = raw

/**
 * Create a middleware to parse raw bodies.
 *
 * @param {object} [options]
 * @return {function}
 * @api public
 */

function raw(options) {
  options = options || {};

  var inflate = options.inflate !== false
  var limit = typeof options.limit !== 'number'
    ? bytes(options.limit || '100kb')
    : options.limit
  var type = options.type || 'application/octet-stream'
  var verify = options.verify || false

  if (verify !== false && typeof verify !== 'function') {
    throw new TypeError('option verify must be function')
  }

  function parse(buf) {
    return buf
  }

  return function rawParser(req, res, next) {
    if (req._body) return next()
    req.body = req.body || {}

    if (!typeis(req, type)) return next()

    // read
    read(req, res, next, parse, {
      encoding: null,
      inflate: inflate,
      limit: limit,
      verify: verify
    })
  }
}
