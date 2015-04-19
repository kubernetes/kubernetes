// API references:
// 
// * http://dev.w3.org/html5/websockets/
// * http://dvcs.w3.org/hg/domcore/raw-file/tip/Overview.html#interface-eventtarget
// * http://dvcs.w3.org/hg/domcore/raw-file/tip/Overview.html#interface-event

var util   = require('util'),
    driver = require('websocket-driver'),
    API    = require('./websocket/api');

var WebSocket = function(request, socket, body, protocols, options) {
  options = options || {};

  this._stream = socket;
  this._driver = driver.http(request, {maxLength: options.maxLength, protocols: protocols});

  var self = this;
  if (!this._stream || !this._stream.writable) return;

  var catchup = function() { self._stream.removeListener('data', catchup) };
  this._stream.on('data', catchup);

  this._driver.io.write(body);
  API.call(this, options);

  process.nextTick(function() {
    self._driver.start();
  });
};
util.inherits(WebSocket, API);

WebSocket.isWebSocket = function(request) {
  return driver.isWebSocket(request);
};

WebSocket.WebSocket   = WebSocket;
WebSocket.Client      = require('./websocket/client');
WebSocket.EventSource = require('./eventsource');

module.exports        = WebSocket;
