'use strict';

// Protocol references:
// 
// * http://tools.ietf.org/html/draft-hixie-thewebsocketprotocol-75
// * http://tools.ietf.org/html/draft-hixie-thewebsocketprotocol-76
// * http://tools.ietf.org/html/draft-ietf-hybi-thewebsocketprotocol-17

var Base   = require('./driver/base'),
    Client = require('./driver/client'),
    Server = require('./driver/server');

var Driver = {
  client: function(url, options) {
    options = options || {};
    if (options.masking === undefined) options.masking = true;
    return new Client(url, options);
  },

  server: function(options) {
    options = options || {};
    if (options.requireMasking === undefined) options.requireMasking = true;
    return new Server(options);
  },

  http: function() {
    return Server.http.apply(Server, arguments);
  },

  isSecureRequest: function(request) {
    return Server.isSecureRequest(request);
  },

  isWebSocket: function(request) {
    if (request.method !== 'GET') return false;

    var connection = request.headers.connection || '',
        upgrade    = request.headers.upgrade || '';

    return request.method === 'GET' &&
           connection.toLowerCase().split(/ *, */).indexOf('upgrade') >= 0 &&
           upgrade.toLowerCase() === 'websocket';
  },

  validateOptions: function(options, validKeys) {
    Base.validateOptions(options, validKeys);
  }
};

module.exports = Driver;
