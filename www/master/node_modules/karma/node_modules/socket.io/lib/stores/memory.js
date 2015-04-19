
/*!
 * socket.io-node
 * Copyright(c) 2011 LearnBoost <dev@learnboost.com>
 * MIT Licensed
 */

/**
 * Module dependencies.
 */

var crypto = require('crypto')
  , Store = require('../store');

/**
 * Exports the constructor.
 */

exports = module.exports = Memory;
Memory.Client = Client;

/**
 * Memory store
 *
 * @api public
 */

function Memory (opts) {
  Store.call(this, opts);
};

/**
 * Inherits from Store.
 */

Memory.prototype.__proto__ = Store.prototype;

/**
 * Publishes a message.
 *
 * @api private
 */

Memory.prototype.publish = function () { };

/**
 * Subscribes to a channel
 *
 * @api private
 */

Memory.prototype.subscribe = function () { };

/**
 * Unsubscribes
 *
 * @api private
 */

Memory.prototype.unsubscribe = function () { };

/**
 * Client constructor
 *
 * @api private
 */

function Client () {
  Store.Client.apply(this, arguments);
  this.data = {};
};

/**
 * Inherits from Store.Client
 */

Client.prototype.__proto__ = Store.Client;

/**
 * Gets a key
 *
 * @api public
 */

Client.prototype.get = function (key, fn) {
  fn(null, this.data[key] === undefined ? null : this.data[key]);
  return this;
};

/**
 * Sets a key
 *
 * @api public
 */

Client.prototype.set = function (key, value, fn) {
  this.data[key] = value;
  fn && fn(null);
  return this;
};

/**
 * Has a key
 *
 * @api public
 */

Client.prototype.has = function (key, fn) {
  fn(null, key in this.data);
};

/**
 * Deletes a key
 *
 * @api public
 */

Client.prototype.del = function (key, fn) {
  delete this.data[key];
  fn && fn(null);
  return this;
};

/**
 * Destroys the client.
 *
 * @param {Number} number of seconds to expire data
 * @api private
 */

Client.prototype.destroy = function (expiration) {
  if ('number' != typeof expiration) {
    this.data = {};
  } else {
    var self = this;

    setTimeout(function () {
      self.data = {};
    }, expiration * 1000);
  }

  return this;
};
