/*!
 * socket.io-node
 * Copyright(c) 2011 LearnBoost <dev@learnboost.com>
 * MIT Licensed
 */

/**
 * Module requirements.
 */
var WebSocket = require('./websocket');

/**
 * Export the constructor.
 */

exports = module.exports = FlashSocket;

/**
 * The FlashSocket transport is just a proxy
 * for WebSocket connections.
 *
 * @api public
 */
 
function FlashSocket (mng, data, req) {
  return WebSocket.call(this, mng, data, req);
}

/**
 * Inherits from WebSocket.
 */

FlashSocket.prototype.__proto__ = WebSocket.prototype;

/**
 * Transport name
 *
 * @api public
 */

FlashSocket.prototype.name = 'flashsocket';

/**
 * Listens for new configuration changes of the Manager
 * this way we can enable and disable the flash server.
 *
 * @param {Manager} Manager instance.
 * @api private
 */


FlashSocket.init = function (manager) {
  var server;
  function create () {

    // Drop out immediately if the user has
    // disabled the flash policy server
    if (!manager.get('flash policy server')) {
      return;
    }

    server = require('policyfile').createServer({ 
      log: function(msg){
        manager.log.info(msg);
      }
    }, manager.get('origins'));

    server.on('close', function (e) {
      server = null;
    });

    server.listen(manager.get('flash policy port'), manager.server);

    manager.flashPolicyServer = server;
  }

  // listen for origin changes, so we can update the server
  manager.on('set:origins', function (value, key) {
    if (!server) return;

    // update the origins and compile a new response buffer
    server.origins = Array.isArray(value) ? value : [value];
    server.compile();
  });

  // destory the server and create a new server
  manager.on('set:flash policy port', function (value, key) {
    var transports = manager.get('transports');
    if (~transports.indexOf('flashsocket')) {
      if (server) {
        if (server.port === value) return;
        // destroy the server and rebuild it on a new port
        try {
          server.close();
        }
        catch (e) { /* ignore exception. could e.g. be that the server isn't started yet */ }
      }
      create();
    }
  });

  // create or destroy the server
  manager.on('set:flash policy server', function (value, key) {
    var transports = manager.get('transports');
    if (~transports.indexOf('flashsocket')) {
      if (server && !value) {
        // destroy the server
        try {
          server.close();
        }
        catch (e) { /* ignore exception. could e.g. be that the server isn't started yet */ }
      }
    } else if (!server && value) {
      // create the server
      create();
    }
  });

  // only start the server
  manager.on('set:transports', function (value, key){
    if (!server && ~manager.get('transports').indexOf('flashsocket')) {
      create();
    }
  });
  // check if we need to initialize at start
  if (~manager.get('transports').indexOf('flashsocket')){
    create();
  }
};
