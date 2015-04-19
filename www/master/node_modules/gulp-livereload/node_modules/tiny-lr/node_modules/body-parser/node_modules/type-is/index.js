
var typer = require('media-typer')
var mime = require('mime-types')

module.exports = typeofrequest;
typeofrequest.is = typeis;
typeofrequest.hasBody = hasbody;
typeofrequest.normalize = normalize;
typeofrequest.match = mimeMatch;

/**
 * Compare a `value` content-type with `types`.
 * Each `type` can be an extension like `html`,
 * a special shortcut like `multipart` or `urlencoded`,
 * or a mime type.
 *
 * If no types match, `false` is returned.
 * Otherwise, the first `type` that matches is returned.
 *
 * @param {String} value
 * @param {Array} types
 * @return String
 */

function typeis(value, types_) {
  var i
  var types = types_

  // remove parameters and normalize
  var val = typenormalize(value)

  // no type or invalid
  if (!val) {
    return false
  }

  // support flattened arguments
  if (types && !Array.isArray(types)) {
    types = new Array(arguments.length - 1)
    for (i = 0; i < types.length; i++) {
      types[i] = arguments[i + 1]
    }
  }

  // no types, return the content type
  if (!types || !types.length) {
    return val
  }

  var type
  for (i = 0; i < types.length; i++) {
    if (mimeMatch(normalize(type = types[i]), val)) {
      return type[0] === '+' || ~type.indexOf('*')
        ? val
        : type
    }
  }

  // no matches
  return false;
}

/**
 * Check if a request has a request body.
 * A request with a body __must__ either have `transfer-encoding`
 * or `content-length` headers set.
 * http://www.w3.org/Protocols/rfc2616/rfc2616-sec4.html#sec4.3
 *
 * @param {Object} request
 * @return {Boolean}
 * @api public
 */

function hasbody(req) {
  var headers = req.headers;
  if ('transfer-encoding' in headers) return true;
  return !isNaN(headers['content-length']);
}

/**
 * Check if the incoming request contains the "Content-Type"
 * header field, and it contains any of the give mime `type`s.
 * If there is no request body, `null` is returned.
 * If there is no content type, `false` is returned.
 * Otherwise, it returns the first `type` that matches.
 *
 * Examples:
 *
 *     // With Content-Type: text/html; charset=utf-8
 *     this.is('html'); // => 'html'
 *     this.is('text/html'); // => 'text/html'
 *     this.is('text/*', 'application/json'); // => 'text/html'
 *
 *     // When Content-Type is application/json
 *     this.is('json', 'urlencoded'); // => 'json'
 *     this.is('application/json'); // => 'application/json'
 *     this.is('html', 'application/*'); // => 'application/json'
 *
 *     this.is('html'); // => false
 *
 * @param {String|Array} types...
 * @return {String|false|null}
 * @api public
 */

function typeofrequest(req, types_) {
  var types = types_

  // no body
  if (!hasbody(req)) {
    return null
  }

  // support flattened arguments
  if (arguments.length > 2) {
    types = new Array(arguments.length - 1)
    for (var i = 0; i < types.length; i++) {
      types[i] = arguments[i + 1]
    }
  }

  // request content type
  var value = req.headers['content-type']

  return typeis(value, types);
}

/**
 * Normalize a mime type.
 * If it's a shorthand, expand it to a valid mime type.
 *
 * In general, you probably want:
 *
 *   var type = is(req, ['urlencoded', 'json', 'multipart']);
 *
 * Then use the appropriate body parsers.
 * These three are the most common request body types
 * and are thus ensured to work.
 *
 * @param {String} type
 * @api private
 */

function normalize(type) {
  switch (type) {
    case 'urlencoded': return 'application/x-www-form-urlencoded';
    case 'multipart':
      type = 'multipart/*';
      break;
  }

  return type[0] === '+' || ~type.indexOf('/')
    ? type
    : mime.lookup(type)
}

/**
 * Check if `exected` mime type
 * matches `actual` mime type with
 * wildcard and +suffix support.
 *
 * @param {String} expected
 * @param {String} actual
 * @return {Boolean}
 * @api private
 */

function mimeMatch(expected, actual) {
  // invalid type
  if (expected === false) {
    return false
  }

  // exact match
  if (expected === actual) {
    return true
  }

  actual = actual.split('/');

  if (expected[0] === '+') {
    // support +suffix
    return Boolean(actual[1])
      && expected.length <= actual[1].length
      && expected === actual[1].substr(0 - expected.length)
  }

  if (!~expected.indexOf('*')) return false;

  expected = expected.split('/');

  if (expected[0] === '*') {
    // support */yyy
    return expected[1] === actual[1]
  }

  if (expected[1] === '*') {
    // support xxx/*
    return expected[0] === actual[0]
  }

  if (expected[1][0] === '*' && expected[1][1] === '+') {
    // support xxx/*+zzz
    return expected[0] === actual[0]
      && expected[1].length <= actual[1].length + 1
      && expected[1].substr(1) === actual[1].substr(1 - expected[1].length)
  }

  return false
}

/**
 * Normalize a type and remove parameters.
 *
 * @param {string} value
 * @return {string}
 * @api private
 */

function typenormalize(value) {
  try {
    var type = typer.parse(value)
    delete type.parameters
    return typer.format(type)
  } catch (err) {
    return null
  }
}
