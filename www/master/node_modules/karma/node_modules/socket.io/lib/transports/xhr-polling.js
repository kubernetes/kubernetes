
/*!
 * socket.io-node
 * Copyright(c) 2011 LearnBoost <dev@learnboost.com>
 * MIT Licensed
 */

/**
 * Module requirements.
 */

var HTTPPolling = require('./http-polling');

/**
 * Export the constructor.
 */

exports = module.exports = XHRPolling;

/**
 * Ajax polling transport.
 *
 * @api public
 */

function XHRPolling (mng, data, req) {
  HTTPPolling.call(this, mng, data, req);
};

/**
 * Inherits from Transport.
 */

XHRPolling.prototype.__proto__ = HTTPPolling.prototype;

/**
 * Transport name
 *
 * @api public
 */

XHRPolling.prototype.name = 'xhr-polling';

/**
 * Frames data prior to write.
 *
 * @api private
 */

XHRPolling.prototype.doWrite = function (data) {
  HTTPPolling.prototype.doWrite.call(this);

  var origin = this.req.headers.origin
    , headers = {
          'Content-Type': 'text/plain; charset=UTF-8'
        , 'Content-Length': data === undefined ? 0 : Buffer.byteLength(data)
        , 'Connection': 'Keep-Alive'
      };

  if (origin) {
    // https://developer.mozilla.org/En/HTTP_Access_Control
    headers['Access-Control-Allow-Origin'] = origin;
    headers['Access-Control-Allow-Credentials'] = 'true';
  }

  this.response.writeHead(200, headers);
  this.response.write(data);
  this.log.debug(this.name + ' writing', data);
};
