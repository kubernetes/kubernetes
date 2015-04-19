
/*!
 * socket.io-node
 * Copyright(c) 2011 LearnBoost <dev@learnboost.com>
 * MIT Licensed
 */

/**
 * Module requirements.
 */

var Transport = require('../transport')
  , parser = require('../parser')
  , qs = require('querystring');

/**
 * Export the constructor.
 */

exports = module.exports = HTTPTransport;

/**
 * HTTP interface constructor. For all non-websocket transports.
 *
 * @api public
 */

function HTTPTransport (mng, data, req) {
  Transport.call(this, mng, data, req);
};

/**
 * Inherits from Transport.
 */

HTTPTransport.prototype.__proto__ = Transport.prototype;

/**
 * Handles a request.
 *
 * @api private
 */

HTTPTransport.prototype.handleRequest = function (req) {

  // Always set the response in case an error is returned to the client
  this.response = req.res;

  if (req.method == 'POST') {
    var buffer = ''
      , res = req.res
      , origin = req.headers.origin
      , headers = { 'Content-Length': 1, 'Content-Type': 'text/plain; charset=UTF-8' }
      , self = this;

    req.on('data', function (data) {
      buffer += data;

      if (Buffer.byteLength(buffer) >= self.manager.get('destroy buffer size')) {
        buffer = '';
        req.connection.destroy();
      }
    });

    req.on('end', function () {
      res.writeHead(200, headers);
      res.end('1');

      self.onData(self.postEncoded ? qs.parse(buffer).d : buffer);
    });

    // prevent memory leaks for uncompleted requests
    req.on('close', function () {
      buffer = '';
      self.onClose();
    });

    if (origin) {
      // https://developer.mozilla.org/En/HTTP_Access_Control
      headers['Access-Control-Allow-Origin'] = origin;
      headers['Access-Control-Allow-Credentials'] = 'true';
      headers['X-XSS-Protection'] = '0';
    }
  } else {
    Transport.prototype.handleRequest.call(this, req);
  }
};

/**
 * Handles data payload.
 *
 * @api private
 */

HTTPTransport.prototype.onData = function (data) {
  var messages = parser.decodePayload(data);
  this.log.debug(this.name + ' received data packet', data);

  for (var i = 0, l = messages.length; i < l; i++) {
    this.onMessage(messages[i]);
  }
};

/**
 * Closes the request-response cycle
 *
 * @api private
 */

HTTPTransport.prototype.doClose = function () {
  this.response.end();
};

/**
 * Writes a payload of messages
 *
 * @api private
 */

HTTPTransport.prototype.payload = function (msgs) {
  this.write(parser.encodePayload(msgs));
};
