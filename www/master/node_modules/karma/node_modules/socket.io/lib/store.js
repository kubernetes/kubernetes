
/*!
 * socket.io-node
 * Copyright(c) 2011 LearnBoost <dev@learnboost.com>
 * MIT Licensed
 */

/**
 * Expose the constructor.
 */

exports = module.exports = Store;

/**
 * Module dependencies.
 */

var EventEmitter = process.EventEmitter;

/**
 * Store interface
 *
 * @api public
 */

function Store (options) {
  this.options = options;
  this.clients = {};
};

/**
 * Inherit from EventEmitter.
 */

Store.prototype.__proto__ = EventEmitter.prototype;

/**
 * Initializes a client store
 *
 * @param {String} id
 * @api public
 */

Store.prototype.client = function (id) {
  if (!this.clients[id]) {
    this.clients[id] = new (this.constructor.Client)(this, id);
  }

  return this.clients[id];
};

/**
 * Destroys a client
 *
 * @api {String} sid
 * @param {Number} number of seconds to expire client data
 * @api private
 */

Store.prototype.destroyClient = function (id, expiration) {
  if (this.clients[id]) {
    this.clients[id].destroy(expiration);
    delete this.clients[id];
  }

  return this;
};

/**
 * Destroys the store
 *
 * @param {Number} number of seconds to expire client data
 * @api private
 */

Store.prototype.destroy = function (clientExpiration) {
  var keys = Object.keys(this.clients)
    , count = keys.length;

  for (var i = 0, l = count; i < l; i++) {
    this.destroyClient(keys[i], clientExpiration);
  }

  this.clients = {};

  return this;
};

/**
 * Client.
 *
 * @api public
 */

Store.Client = function (store, id) {
  this.store = store;
  this.id = id;
};
