
/**
 * Module dependencies.
 */

var url = require('url');
var LRU = require('lru-cache');
var HttpProxyAgent = require('http-proxy-agent');
var HttpsProxyAgent = require('https-proxy-agent');
var SocksProxyAgent = require('socks-proxy-agent');

/**
 * Module exports.
 */

exports = module.exports = proxy;

/**
 * Number of `http.Agent` instances to cache.
 *
 * This value was arbitrarily chosen... a better
 * value could be conceived with some benchmarks.
 */

var cacheSize = 20;

/**
 * Cache for `http.Agent` instances.
 */

exports.cache = new LRU(cacheSize);

/**
 * The built-in proxy types.
 */

exports.proxies = {
  http: httpOrHttpsProxy,
  https: httpOrHttpsProxy,
  socks: socksProxy
};

/**
 * Attempts to get an `http.Agent` instance based off of the given proxy URI
 * information, and the `secure` flag.
 *
 * An LRU cache is used, to prevent unnecessary creation of proxy
 * `http.Agent` instances.
 *
 * @param {String} uri proxy url
 * @param {Boolean} secure true if this is for an HTTPS request, false for HTTP
 * @return {http.Agent}
 * @api public
 */

function proxy (uri, secure) {

  // parse the URI into an opts object if it's a String
  var proxyParsed = uri;
  if ('string' == typeof uri) {
    proxyParsed = url.parse(uri);
  }

  // get the requested proxy "protocol"
  var protocol = proxyParsed.protocol;
  if (!protocol) {
    var types = [];
    for (var type in proxies) types.push(type);
    throw new TypeError('you must specify a string "protocol" for the proxy type (' + types.join(', ') + ')');
  }

  // strip the trailing ":" if present
  if (':' == protocol[protocol.length - 1]) {
    protocol = protocol.substring(0, protocol.length - 1);
  }

  // get the proxy `http.Agent` creation function
  var proxyFn = exports.proxies[protocol];
  if ('function' != typeof proxyFn) {
    throw new TypeError('unsupported proxy protocol: "' + protocol + '"');
  }

  // format the proxy info back into a URI, since an opts object
  // could have been passed in originally. This generated URI is used
  // as part of the "key" for the LRU cache
  var proxyUri = url.format({
    protocol: protocol + ':',
    slashes: true,
    hostname: proxyParsed.hostname || proxyParsed.host,
    port: proxyParsed.port
  });

  // create the "key" for the LRU cache
  var key = proxyUri;
  if (secure) key += ' secure';

  // attempt to get a cached `http.Agent` instance first
  var agent = exports.cache.get(key);
  if (!agent) {
    // get an `http.Agent` instance from protocol-specific agent function
    agent = proxyFn(proxyParsed, secure);
    if (agent) exports.cache.set(key, agent);
  } else {
    //console.error('cache hit! %j', key);
  }

  return agent;
}

/**
 * Default "http" and "https" proxy URI handlers.
 *
 * @api protected
 */

function httpOrHttpsProxy (proxy, secure) {
  if (secure) {
    // HTTPS
    return new HttpsProxyAgent(proxy);
  } else {
    // HTTP
    return new HttpProxyAgent(proxy);
  }
}

/**
 * Default "socks" proxy URI handler.
 *
 * @api protected
 */

function socksProxy (proxy, secure) {
  return new SocksProxyAgent(proxy, secure);
}
