/*
 * ws.js: Macros for proxying Websocket requests
 *
 * (C) 2010 Nodejitsu Inc.
 * MIT LICENCE
 *
 */

var assert = require('assert'),
    fs = require('fs'),
    async = require('async'),
    io = require('socket.io-client'),
    WebSocket = require('ws'),
    helpers = require('../helpers/index');

//
// ### function assertSendRecieve (options)
// #### @options {Object} Options for creating this assertion.
// ####    @raw    {boolean} Enables raw `ws.WebSocket`.
// ####    @uri    {string}  URI of the proxy server.
// ####    @input  {string}  Input to assert sent to the target ws server.
// ####    @output {string}  Output to assert from the taget ws server.
//
// Creates a `socket.io` or raw `WebSocket` connection and asserts that
// `options.input` is sent to and `options.output` is received from the
// connection.
//
exports.assertSendReceive = function (options) {
  if (!options.raw) {
    return {
      topic: function () {
        var socket = io.connect(options.uri);
        socket.on('outgoing', this.callback.bind(this, null));
        socket.emit('incoming', options.input);
      },
      "should send input and receive output": function (_, data) {
        assert.equal(data, options.output);
      }
    };
  }

  return {
    topic: function () {
      var socket = new WebSocket(options.uri);
      socket.on('message', this.callback.bind(this, null));
      socket.on('open', function () {
        socket.send(options.input);
      });
    },
    "should send input and recieve output": function (_, data, flags) {
      assert.equal(data, options.output);
    }
  };
};

//
// ### function assertProxied (options)
// #### @options {Object} Options for this test
// ####    @latency {number}  Latency in milliseconds for the proxy server.
// ####    @ports   {Object}  Ports for the request (target, proxy).
// ####    @input   {string}  Input to assert sent to the target ws server.
// ####    @output  {string}  Output to assert from the taget ws server.
// ####    @raw     {boolean} Enables raw `ws.Server` usage.
//
// Creates a complete end-to-end test for requesting against an
// http proxy.
//
exports.assertProxied = function (options) {
  options = options || {};

  var ports    = options.ports    || helpers.nextPortPair,
      input    = options.input    || 'hello world to ' + ports.target,
      output   = options.output   || 'hello world from ' + ports.target,
      protocol = helpers.protocols.proxy;

  if (options.raw) {
    protocol = helpers.protocols.proxy === 'https'
      ? 'wss'
      : 'ws';
  }

  return {
    topic: function () {
      helpers.ws.createServerPair({
        target: {
          input: input,
          output: output,
          port: ports.target,
          raw: options.raw
        },
        proxy: {
          latency: options.latency,
          port: ports.proxy,
          proxy: {
            target: {
              https: helpers.protocols.target === 'https',
              host: '127.0.0.1',
              port: ports.target
            }
          }
        }
      }, this.callback);
    },
    "the proxy Websocket connection": exports.assertSendReceive({
      uri: protocol + '://127.0.0.1:' + ports.proxy,
      input: input,
      output: output,
      raw: options.raw
    })
  };
};

//
// ### function assertProxiedtoRoutes (options, nested)
// #### @options {Object} Options for this ProxyTable-based test
// ####    @raw          {boolean}       Enables ws.Server usage.
// ####    @routes       {Object|string} Routes to use for the proxy.
// ####    @hostnameOnly {boolean}       Enables hostnameOnly routing.
// #### @nested  {Object} Nested vows to add to the returned context.
//
// Creates a complete end-to-end test for requesting against an
// http proxy using `options.routes`:
//
// 1. Creates target servers for all routes in `options.routes.`
// 2. Creates a proxy server.
// 3. Ensure Websocket connections to the proxy server for all route targets
//    can send input and recieve output.
//
exports.assertProxiedToRoutes = function (options, nested) {
  //
  // Assign dynamic ports to the routes to use.
  //
  options.routes = helpers.http.assignPortsToRoutes(options.routes);

  //
  // Parse locations from routes for making assertion requests.
  //
  var locations = helpers.http.parseRoutes(options),
      protocol = helpers.protocols.proxy,
      port = helpers.nextPort,
      context,
      proxy;

  if (options.raw) {
    protocol = helpers.protocols.proxy === 'https'
      ? 'wss'
      : 'ws';
  }

  if (options.filename) {
    //
    // If we've been passed a filename write the routes to it
    // and setup the proxy options to use that file.
    //
    fs.writeFileSync(options.filename, JSON.stringify({ router: options.routes }));
    proxy = { router: options.filename };
  }
  else {
    //
    // Otherwise just use the routes themselves.
    //
    proxy = {
      hostnameOnly: options.hostnameOnly,
      router: options.routes
    };
  }

  //
  // Create the test context which creates all target
  // servers for all routes and a proxy server.
  //
  context = {
    topic: function () {
      var that = this;

      async.waterfall([
        //
        // 1. Create all the target servers
        //
        async.apply(
          async.forEach,
          locations,
          function createRouteTarget(location, next) {
            helpers.ws.createServer({
              raw: options.raw,
              port: location.target.port,
              output: 'hello from ' + location.source.href,
              input: 'hello to ' + location.source.href
            }, next);
          }
        ),
        //
        // 2. Create the proxy server
        //
        async.apply(
          helpers.http.createProxyServer,
          {
            port: port,
            latency: options.latency,
            routing: true,
            proxy: proxy
          }
        )
      ], function (_, server) {
        //
        // 3. Set the proxy server for later use
        //
        that.proxyServer = server;
        that.callback();
      });

      //
      // 4. Assign the port to the context for later use
      //
      this.port = port;
    }
  };

  //
  // Add test assertions for each of the route locations.
  //
  locations.forEach(function (location) {
    context[location.source.href] = exports.assertSendRecieve({
      uri: protocol + '://127.0.0.1:' + port + location.source.path,
      output: 'hello from ' + location.source.href,
      input: 'hello to ' + location.source.href,
      raw: options.raw
    });
  });

  return context;
};