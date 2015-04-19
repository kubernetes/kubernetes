/*!
 * body-parser
 * Copyright(c) 2014 Jonathan Ong
 * Copyright(c) 2014 Douglas Christopher Wilson
 * MIT Licensed
 */

/**
 * Module dependencies.
 */

var bytes = require('bytes')
var read = require('../read')
var typer = require('media-typer')
var typeis = require('type-is')

/**
 * Module exports.
 */

module.exports = json

/**
 * RegExp to match the first non-space in a string.
 */

var firstcharRegExp = /^\s*(.)/

/**
 * Create a middleware to parse JSON bodies.
 *
 * @param {object} [options]
 * @return {function}
 * @api public
 */

function json(options) {
  options = options || {}

  var limit = typeof options.limit !== 'number'
    ? bytes(options.limit || '100kb')
    : options.limit
  var inflate = options.inflate !== false
  var reviver = options.reviver
  var strict = options.strict !== false
  var type = options.type || 'json'
  var verify = options.verify || false

  if (verify !== false && typeof verify !== 'function') {
    throw new TypeError('option verify must be function')
  }

  function parse(body) {
    if (body.length === 0) {
      // special-case empty json body, as it's a common client-side mistake
      // TODO: maybe make this configurable or part of "strict" option
      return {}
    }

    if (strict) {
      var first = firstchar(body)

      if (first !== '{' && first !== '[') {
        throw new Error('invalid json')
      }
    }

    return JSON.parse(body, reviver)
  }

  return function jsonParser(req, res, next) {
    if (req._body) return next()
    req.body = req.body || {}

    if (!typeis(req, type)) return next()

    // RFC 7159 sec 8.1
    var charset = typer.parse(req).parameters.charset || 'utf-8'
    if (charset.substr(0, 4).toLowerCase() !== 'utf-') {
      var err = new Error('unsupported charset')
      err.status = 415
      next(err)
      return
    }

    // read
    read(req, res, next, parse, {
      encoding: charset,
      inflate: inflate,
      limit: limit,
      verify: verify
    })
  }
}

/**
 * Get the first non-whitespace character in a string.
 *
 * @param {string} str
 * @return {function}
 * @api public
 */


function firstchar(str) {
  var match = firstcharRegExp.exec(str)
  return match ? match[1] : ''
}
