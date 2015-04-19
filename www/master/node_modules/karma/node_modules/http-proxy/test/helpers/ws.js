/*
 * ws.js: Top level include for node-http-proxy websocket helpers
 *
 * (C) 2010 Nodejitsu Inc.
 * MIT LICENCE
 *
 */

var assert = require('assert'),
    https = require('https'),
    async = require('async'),
    io = require('socket.io'),
    ws = require('ws'),
    helpers = require('./index'),
    protocols = helpers.protocols,
    http = require('./http');

//
// ### function createServerPair (options, callback)
// #### @options {Object} Options to create target and proxy server.
// ####    @target {Object} Options for the target server.
// ####    @proxy  {Object} Options for the proxy server.
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
      http.createProxyServer(options.proxy, next);
    }
  ], callback);
};

//
// ### function createServer (options, callback)
// #### @options {Object} Options for creating the socket.io or ws server.
// ####    @raw   {boolean} Enables ws.Websocket server.
//
// Creates a socket.io or ws server using the specified `options`.
//
exports.createServer = function (options, callback) {
  return options.raw
    ? exports.createWsServer(options, callback)
    : exports.createSocketIoServer(options, callback);
};

//
// ### function createSocketIoServer (options, callback)
// #### @options {Object} Options for creating the socket.io server
// ####    @port   {number} Port to listen on
// ####    @input  {string} Input to expect from the only socket
// ####    @output {string} Output to send the only socket
//
// Creates a socket.io server on the specified `options.port` which
// will expect `options.input` and then send `options.output`.
//
exports.createSocketIoServer = function (options, callback) {
  var server = protocols.target === 'https'
    ? io.listen(options.port, helpers.https, callback)
    : io.listen(options.port, callback);

  server.sockets.on('connection', function (socket) {
    socket.on('incoming', function (data) {
      assert.equal(data, options.input);
      socket.emit('outgoing', options.output);
    });
  });
};

//
// ### function createWsServer (options, callback)
// #### @options {Object} Options for creating the ws.Server instance
// ####    @port   {number} Port to listen on
// ####    @input  {string} Input to expect from the only socket
// ####    @output {string} Output to send the only socket
//
// Creates a ws.Server instance on the specified `options.port` which
// will expect `options.input` and then send `options.output`.
//
exports.createWsServer = function (options, callback) {
  var server,
      wss;

  if (protocols.target === 'https') {
    server = https.createServer(helpers.https, function (req, res) {
      req.writeHead(200);
      req.end();
    }).listen(options.port, callback);

    wss = new ws.Server({ server: server });
  }
  else {
    wss = new ws.Server({ port: options.port }, callback);
  }

  wss.on('connection', function (socket) {
    socket.on('message', function (data) {
      assert.equal(data, options.input);
      socket.send(options.output);
    });
  });
};