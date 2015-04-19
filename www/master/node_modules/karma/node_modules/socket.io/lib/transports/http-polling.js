
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
 * Exports the constructor.
 */

exports = module.exports = HTTPPolling;

/**
 * HTTP polling constructor.
 *
 * @api public.
 */

function HTTPPolling (mng, data, req) {
  HTTPTransport.call(this, mng, data, req);
};

/**
 * Inherits from HTTPTransport.
 *
 * @api public.
 */

HTTPPolling.prototype.__proto__ = HTTPTransport.prototype;

/**
 * Transport name
 *
 * @api public
 */

HTTPPolling.prototype.name = 'httppolling';

/**
 * Override setHandlers
 *
 * @api private
 */

HTTPPolling.prototype.setHandlers = function () {
  HTTPTransport.prototype.setHandlers.call(this);
  this.socket.removeListener('end', this.bound.end);
  this.socket.removeListener('close', this.bound.close);
};

/**
 * Removes heartbeat timeouts for polling.
 */

HTTPPolling.prototype.setHeartbeatInterval = function () {
  return this;
};

/**
 * Handles a request
 *
 * @api private
 */

HTTPPolling.prototype.handleRequest = function (req) {
  HTTPTransport.prototype.handleRequest.call(this, req);

  if (req.method == 'GET') {
    var self = this;

    this.pollTimeout = setTimeout(function () {
      self.packet({ type: 'noop' });
      self.log.debug(self.name + ' closed due to exceeded duration');
    }, this.manager.get('polling duration') * 1000);

    this.log.debug('setting poll timeout');
  }
};

/**
 * Clears polling timeout
 *
 * @api private
 */

HTTPPolling.prototype.clearPollTimeout = function () {
  if (this.pollTimeout) {
    clearTimeout(this.pollTimeout);
    this.pollTimeout = null;
    this.log.debug('clearing poll timeout');
  }

  return this;
};

/**
 * Override clear timeouts to clear the poll timeout
 *
 * @api private
 */

HTTPPolling.prototype.clearTimeouts = function () {
  HTTPTransport.prototype.clearTimeouts.call(this);

  this.clearPollTimeout();
};

/**
 * doWrite to clear poll timeout
 *
 * @api private
 */

HTTPPolling.prototype.doWrite = function () {
  this.clearPollTimeout();
};

/**
 * Performs a write.
 *
 * @api private.
 */

HTTPPolling.prototype.write = function (data, close) {
  this.doWrite(data);
  this.response.end();
  this.onClose();
};

/**
 * Override end.
 *
 * @api private
 */

HTTPPolling.prototype.end = function (reason) {
  this.clearPollTimeout();
  return HTTPTransport.prototype.end.call(this, reason);
};

