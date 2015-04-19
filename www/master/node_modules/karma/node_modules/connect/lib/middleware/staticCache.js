
/*!
 * Connect - staticCache
 * Copyright(c) 2011 Sencha Inc.
 * MIT Licensed
 */

/**
 * Module dependencies.
 */

var utils = require('../utils')
  , Cache = require('../cache')
  , fresh = require('fresh');

/**
 * Static cache:
 *
 * Enables a memory cache layer on top of
 * the `static()` middleware, serving popular
 * static files.
 *
 * By default a maximum of 128 objects are
 * held in cache, with a max of 256k each,
 * totalling ~32mb.
 *
 * A Least-Recently-Used (LRU) cache algo
 * is implemented through the `Cache` object,
 * simply rotating cache objects as they are
 * hit. This means that increasingly popular
 * objects maintain their positions while
 * others get shoved out of the stack and
 * garbage collected.
 *
 * Benchmarks:
 *
 *     static(): 2700 rps
 *     node-static: 5300 rps
 *     static() + staticCache(): 7500 rps
 *
 * Options:
 *
 *   - `maxObjects`  max cache objects [128]
 *   - `maxLength`  max cache object length 256kb
 *
 * @param {Object} options
 * @return {Function}
 * @api public
 */

module.exports = function staticCache(options){
  var options = options || {}
    , cache = new Cache(options.maxObjects || 128)
    , maxlen = options.maxLength || 1024 * 256;

  console.warn('connect.staticCache() is deprecated and will be removed in 3.0');
  console.warn('use varnish or similar reverse proxy caches.');

  return function staticCache(req, res, next){
    var key = cacheKey(req)
      , ranges = req.headers.range
      , hasCookies = req.headers.cookie
      , hit = cache.get(key);

    // cache static
    // TODO: change from staticCache() -> cache()
    // and make this work for any request
    req.on('static', function(stream){
      var headers = res._headers
        , cc = utils.parseCacheControl(headers['cache-control'] || '')
        , contentLength = headers['content-length']
        , hit;

      // dont cache set-cookie responses
      if (headers['set-cookie']) return hasCookies = true;

      // dont cache when cookies are present
      if (hasCookies) return;

      // ignore larger files
      if (!contentLength || contentLength > maxlen) return;

      // don't cache partial files
      if (headers['content-range']) return;

      // dont cache items we shouldn't be
      // TODO: real support for must-revalidate / no-cache
      if ( cc['no-cache']
        || cc['no-store']
        || cc['private']
        || cc['must-revalidate']) return;

      // if already in cache then validate
      if (hit = cache.get(key)){
        if (headers.etag == hit[0].etag) {
          hit[0].date = new Date;
          return;
        } else {
          cache.remove(key);
        }
      }

      // validation notifiactions don't contain a steam
      if (null == stream) return;

      // add the cache object
      var arr = [];

      // store the chunks
      stream.on('data', function(chunk){
        arr.push(chunk);
      });

      // flag it as complete
      stream.on('end', function(){
        var cacheEntry = cache.add(key);
        delete headers['x-cache']; // Clean up (TODO: others)
        cacheEntry.push(200);
        cacheEntry.push(headers);
        cacheEntry.push.apply(cacheEntry, arr);
      });
    });

    if (req.method == 'GET' || req.method == 'HEAD') {
      if (ranges) {
        next();
      } else if (!hasCookies && hit && !mustRevalidate(req, hit)) {
        res.setHeader('X-Cache', 'HIT');
        respondFromCache(req, res, hit);
      } else {
        res.setHeader('X-Cache', 'MISS');
        next();
      }
    } else {
      next();
    }
  }
};

/**
 * Respond with the provided cached value.
 * TODO: Assume 200 code, that's iffy.
 *
 * @param {Object} req
 * @param {Object} res
 * @param {Object} cacheEntry
 * @return {String}
 * @api private
 */

function respondFromCache(req, res, cacheEntry) {
  var status = cacheEntry[0]
    , headers = utils.merge({}, cacheEntry[1])
    , content = cacheEntry.slice(2);

  headers.age = (new Date - new Date(headers.date)) / 1000 || 0;

  switch (req.method) {
    case 'HEAD':
      res.writeHead(status, headers);
      res.end();
      break;
    case 'GET':
      if (utils.conditionalGET(req) && fresh(req.headers, headers)) {
        headers['content-length'] = 0;
        res.writeHead(304, headers);
        res.end();
      } else {
        res.writeHead(status, headers);

        function write() {
          while (content.length) {
            if (false === res.write(content.shift())) {
              res.once('drain', write);
              return;
            }
          }
          res.end();
        }

        write();
      }
      break;
    default:
      // This should never happen.
      res.writeHead(500, '');
      res.end();
  }
}

/**
 * Determine whether or not a cached value must be revalidated.
 *
 * @param {Object} req
 * @param {Object} cacheEntry
 * @return {String}
 * @api private
 */

function mustRevalidate(req, cacheEntry) {
  var cacheHeaders = cacheEntry[1]
    , reqCC = utils.parseCacheControl(req.headers['cache-control'] || '')
    , cacheCC = utils.parseCacheControl(cacheHeaders['cache-control'] || '')
    , cacheAge = (new Date - new Date(cacheHeaders.date)) / 1000 || 0;

  if ( cacheCC['no-cache']
    || cacheCC['must-revalidate']
    || cacheCC['proxy-revalidate']) return true;

  if (reqCC['no-cache']) return true;

  if (null != reqCC['max-age']) return reqCC['max-age'] < cacheAge;

  if (null != cacheCC['max-age']) return cacheCC['max-age'] < cacheAge;

  return false;
}

/**
 * The key to use in the cache. For now, this is the URL path and query.
 *
 * 'http://example.com?key=value' -> '/?key=value'
 *
 * @param {Object} req
 * @return {String}
 * @api private
 */

function cacheKey(req) {
  return utils.parseUrl(req).path;
}
