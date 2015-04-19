
/*!
 * socket.io-node
 * Copyright(c) 2011 LearnBoost <dev@learnboost.com>
 * MIT Licensed
 */

/**
 * Module requirements.
 */

var HTTPTransport = require('./http');

/**
 * Export the constructor.
 */

exports = module.exports = HTMLFile;

/**
 * HTMLFile transport constructor.
 *
 * @api public
 */

function HTMLFile (mng, data, req) {
  HTTPTransport.call(this, mng, data, req);
};

/**
 * Inherits from Transport.
 */

HTMLFile.prototype.__proto__ = HTTPTransport.prototype;

/**
 * Transport name
 *
 * @api public
 */

HTMLFile.prototype.name = 'htmlfile';

/**
 * Handles the request.
 *
 * @api private
 */

HTMLFile.prototype.handleRequest = function (req) {
  HTTPTransport.prototype.handleRequest.call(this, req);

  if (req.method == 'GET') {
    req.res.writeHead(200, {
        'Content-Type': 'text/html; charset=UTF-8'
      , 'Connection': 'keep-alive'
      , 'Transfer-Encoding': 'chunked'
    });

    req.res.write(
        '<html><body>'
      + '<script>var _ = function (msg) { parent.s._(msg, document); };</script>'
      + new Array(174).join(' ')
    );
  }
};

/**
 * Performs the write.
 *
 * @api private
 */

HTMLFile.prototype.write = function (data) {
  // escape all forward slashes. see GH-1251
  data = '<script>_(' + JSON.stringify(data).replace(/\//g, '\\/') + ');</script>';

  if (this.response.write(data)) {
    this.drained = true;
  }

  this.log.debug(this.name + ' writing', data);
};
