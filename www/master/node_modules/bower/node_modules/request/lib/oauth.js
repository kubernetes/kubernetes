'use strict'

var querystring = require('querystring')
  , qs = require('qs')
  , caseless = require('caseless')
  , uuid = require('node-uuid')
  , oauth = require('oauth-sign')


exports.buildParams = function (_oauth, uri, method, query, form, qsLib) {
  var oa = {}
  for (var i in _oauth) {
    oa['oauth_' + i] = _oauth[i]
  }
  if (!oa.oauth_version) {
    oa.oauth_version = '1.0'
  }
  if (!oa.oauth_timestamp) {
    oa.oauth_timestamp = Math.floor( Date.now() / 1000 ).toString()
  }
  if (!oa.oauth_nonce) {
    oa.oauth_nonce = uuid().replace(/-/g, '')
  }
  if (!oa.oauth_signature_method) {
    oa.oauth_signature_method = 'HMAC-SHA1'
  }

  var consumer_secret_or_private_key = oa.oauth_consumer_secret || oa.oauth_private_key
  delete oa.oauth_consumer_secret
  delete oa.oauth_private_key

  var token_secret = oa.oauth_token_secret
  delete oa.oauth_token_secret

  var realm = oa.oauth_realm
  delete oa.oauth_realm
  delete oa.oauth_transport_method

  var baseurl = uri.protocol + '//' + uri.host + uri.pathname
  var params = qsLib.parse([].concat(query, form, qsLib.stringify(oa)).join('&'))

  oa.oauth_signature = oauth.sign(
    oa.oauth_signature_method,
    method,
    baseurl,
    params,
    consumer_secret_or_private_key,
    token_secret)

  if (realm) {
    oa.realm = realm
  }

  return oa
}

exports.concatParams = function (oa, sep, wrap) {
  wrap = wrap || ''

  var params = Object.keys(oa).filter(function (i) {
    return i !== 'realm' && i !== 'oauth_signature'
  }).sort()

  if (oa.realm) {
    params.splice(0, 1, 'realm')
  }
  params.push('oauth_signature')

  return params.map(function (i) {
    return i + '=' + wrap + oauth.rfc3986(oa[i]) + wrap
  }).join(sep)
}

exports.oauth = function (args) {
  var uri = args.uri || {}
    , method = args.method || ''
    , headers = caseless(args.headers)
    , body = args.body || ''
    , _oauth = args.oauth || {}
    , qsLib = args.qsLib || qs

  var form
    , query
    , contentType = headers.get('content-type') || ''
    , formContentType = 'application/x-www-form-urlencoded'
    , transport = _oauth.transport_method || 'header'

  if (contentType.slice(0, formContentType.length) === formContentType) {
    contentType = formContentType
    form = body
  }
  if (uri.query) {
    query = uri.query
  }
  if (transport === 'body' && (method !== 'POST' || contentType !== formContentType)) {
    throw new Error('oauth: transport_method of \'body\' requires \'POST\' ' +
      'and content-type \'' + formContentType + '\'')
  }

  var oa = this.buildParams(_oauth, uri, method, query, form, qsLib)

  var data
  switch (transport) {
    case 'header':
      data = 'OAuth ' + this.concatParams(oa, ',', '"')
      break

    case 'query':
      data = (query ? '&' : '?') + this.concatParams(oa, '&')
      break

    case 'body':
      data = (form ? form + '&' : '') + this.concatParams(oa, '&')
      break

    default:
      throw new Error('oauth: transport_method invalid')
  }

  return {oauth:data, transport:transport}
}
