/*
 * http.js: Top level include for node-http-proxy http helpers
 *
 * (C) 2010 Nodejitsu Inc.
 * MIT LICENCE
 *
 */

var assert = require('assert'),
    http = require('http'),
    https = require('https'),
    url = require('url'),
    async = require('async'),
    helpers = require('./index'),
    protocols = helpers.protocols,
    httpProxy = require('../../lib/node-http-proxy');

//
// ### function createServerPair (options, callback)
// #### @options {Object} Options to create target and proxy server.
// #### @callback {function} Continuation to respond to when complete.
//
// Creates http target and proxy servers
//
exports.createServerPair = function (options, callback) {
  async.series([
    //
    // 1. Create the target server
    //
    function createTarget(next) {
      exports.createServer(options.target, next);
    },
    //
    // 2. Create the proxy server
    //
    function createTarget(next) {
      exports.createProxyServer(options.proxy, next);
    }
  ], callback);
};

//
// ### function createServer (options, callback)
// #### @options {Object} Options for creatig an http server.
// ####    @port    {number} Port to listen on
// ####    @output  {string} String to write to each HTTP response
// ####    @headers {Object} Headers to assert are sent by `node-http-proxy`.
// #### @callback {function} Continuation to respond to when complete.
//
// Creates a target server that the tests will proxy to.
//
exports.createServer = function (options, callback) {
  //
  // Request handler to use in either `http`
  // or `https` server.
  //
  function requestHandler(req, res) {
    if (options.headers) {
      Object.keys(options.headers).forEach(function (key) {
        assert.equal(req.headers[key], options.headers[key]);
      });
    }

    if (options.outputHeaders) {
      Object.keys(options.outputHeaders).forEach(function (header) {
        res.setHeader(header, options.outputHeaders[header]);
      });
    }

    setTimeout(function() {
      res.writeHead(200, { 'Content-Type': 'text/plain' });
      res.write(options.output || 'hello proxy'); 
      res.end(); 
    }, options.latency || 1);
  }

  var server = protocols.target === 'https'
    ? https.createServer(helpers.https, requestHandler)
    : http.createServer(requestHandler);

  server.listen(options.port, function () {
    callback(null, this);
  });
};

//
// ### function createProxyServer (options, callback)
// #### @options {Object} Options for creatig an http server.
// ####    @port    {number}  Port to listen on
// ####    @latency {number}  Latency of this server in milliseconds
// ####    @proxy   {Object}  Options to pass to the HttpProxy.
// ####    @routing {boolean} Enables `httpProxy.RoutingProxy`
// #### @callback {function} Continuation to respond to when complete.
//
// Creates a proxy server that the tests will request against.
//
exports.createProxyServer = function (options, callback) {
  if (!options.latency) {
    if (protocols.proxy === 'https') {
      options.proxy.https = helpers.https;
    }
    options.proxy.rejectUnauthorized = false;

    return httpProxy
      .createServer(options.proxy)
      .listen(options.port, function () {
        callback(null, this);
      });
  }

  var server,
      proxy;

  proxy = options.routing
    ? new httpProxy.RoutingProxy(options.proxy)
    : new httpProxy.HttpProxy(options.proxy);

  //
  // Request handler to use in either `http`
  // or `https` server.
  //
  function requestHandler(req, res) {
    var buffer = httpProxy.buffer(req);

    if (options.outputHeaders) {
      Object.keys(options.outputHeaders).forEach(function (header) {
        res.setHeader(header, options.outputHeaders[header]);
      });
    }
    setTimeout(function () {
      //
      // Setup options dynamically for `RoutingProxy.prototype.proxyRequest`
      // or `HttpProxy.prototype.proxyRequest`.
      //
      buffer = options.routing ? { buffer: buffer } : buffer;
      proxy.proxyRequest(req, res, buffer);
    }, options.latency);
  }

  server = protocols.proxy === 'https'
    ? https.createServer(helpers.https, requestHandler)
    : http.createServer(requestHandler);

  server.listen(options.port, function () {
    callback(null, this);
  });
};

//
// ### function assignPortsToRoutes (routes)
// #### @routes {Object} Routing table to assign ports to
//
// Assigns dynamic ports to the `routes` for runtime testing.
//
exports.assignPortsToRoutes = function (routes) {
  Object.keys(routes).forEach(function (source) {
    routes[source] = routes[source].replace('{PORT}', helpers.nextPort);
  });

  return routes;
};

//
// ### function parseRoutes (options)
// #### @options {Object} Options to use when parsing routes
// ####    @protocol {string} Protocol to use in the routes
// ####    @routes   {Object} Routes to parse.
//
// Returns an Array of fully-parsed URLs for the source and
// target of `options.routes`.
//
exports.parseRoutes = function (options) {
  var protocol = options.protocol || 'http',
      routes = options.routes;

  return Object.keys(routes).map(function (source) {
    return {
      source: url.parse(protocol + '://' + source),
      target: url.parse(protocol + '://' + routes[source])
    };
  });
};
