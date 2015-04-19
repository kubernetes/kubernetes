var crypto = require('crypto')
  , qs = require('querystring')
  ;

function sha1 (key, body) {
  return crypto.createHmac('sha1', key).update(body).digest('base64')
}

function rfc3986 (str) {
  return encodeURIComponent(str)
    .replace(/!/g,'%21')
    .replace(/\*/g,'%2A')
    .replace(/\(/g,'%28')
    .replace(/\)/g,'%29')
    .replace(/'/g,'%27')
    ;
}

function hmacsign (httpMethod, base_uri, params, consumer_secret, token_secret) {
  // adapted from https://dev.twitter.com/docs/auth/oauth and 
  // https://dev.twitter.com/docs/auth/creating-signature

  var querystring = Object.keys(params).sort().map(function(key){
    // big WTF here with the escape + encoding but it's what twitter wants
    return escape(rfc3986(key)) + "%3D" + escape(rfc3986(params[key]))
  }).join('%26')

  var base = [
    httpMethod ? httpMethod.toUpperCase() : 'GET',
    rfc3986(base_uri),
    querystring
  ].join('&')

  var key = [
    consumer_secret,
    token_secret || ''
  ].map(rfc3986).join('&')

  return sha1(key, base)
}

exports.hmacsign = hmacsign
exports.rfc3986 = rfc3986
