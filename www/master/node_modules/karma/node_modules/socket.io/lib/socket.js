
/*!
 * socket.io-node
 * Copyright(c) 2011 LearnBoost <dev@learnboost.com>
 * MIT Licensed
 */

/**
 * Module dependencies.
 */

var parser = require('./parser')
  , util = require('./util')
  , EventEmitter = process.EventEmitter

/**
 * Export the constructor.
 */

exports = module.exports = Socket;

/**
 * Default error event listener to prevent uncaught exceptions.
 */

var defaultError = function () {};

/**
 * Socket constructor.
 *
 * @param {Manager} manager instance
 * @param {String} session id
 * @param {Namespace} namespace the socket belongs to
 * @param {Boolean} whether the 
 * @api public
 */

function Socket (manager, id, nsp, readable) {
  this.id = id;
  this.namespace = nsp;
  this.manager = manager;
  this.disconnected = false;
  this.ackPackets = 0;
  this.acks = {};
  this.setFlags();
  this.readable = readable;
  this.store = this.manager.store.client(this.id);
  this.on('error', defaultError);
};

/**
 * Inherits from EventEmitter.
 */

Socket.prototype.__proto__ = EventEmitter.prototype;

/**
 * Accessor shortcut for the handshake data
 *
 * @api private
 */

Socket.prototype.__defineGetter__('handshake', function () {
  return this.manager.handshaken[this.id];
});

/**
 * Accessor shortcut for the transport type
 *
 * @api private
 */

Socket.prototype.__defineGetter__('transport', function () {
  return this.manager.transports[this.id].name;
});

/**
 * Accessor shortcut for the logger.
 *
 * @api private
 */

Socket.prototype.__defineGetter__('log', function () {
  return this.manager.log;
});

/**
 * JSON message flag.
 *
 * @api public
 */

Socket.prototype.__defineGetter__('json', function () {
  this.flags.json = true;
  return this;
});

/**
 * Volatile message flag.
 *
 * @api public
 */

Socket.prototype.__defineGetter__('volatile', function () {
  this.flags.volatile = true;
  return this;
});

/**
 * Broadcast message flag.
 *
 * @api public
 */

Socket.prototype.__defineGetter__('broadcast', function () {
  this.flags.broadcast = true;
  return this;
});

/**
 * Overrides the room to broadcast messages to (flag)
 *
 * @api public
 */

Socket.prototype.to = Socket.prototype.in = function (room) {
  this.flags.room = room;
  return this;
};

/**
 * Resets flags
 *
 * @api private
 */

Socket.prototype.setFlags = function () {
  this.flags = {
      endpoint: this.namespace.name
    , room: ''
  };
  return this;
};

/**
 * Triggered on disconnect
 *
 * @api private
 */

Socket.prototype.onDisconnect = function (reason) {
  if (!this.disconnected) {
    this.$emit('disconnect', reason);
    this.disconnected = true;
  }
};

/**
 * Joins a user to a room.
 *
 * @api public
 */

Socket.prototype.join = function (name, fn) {
  var nsp = this.namespace.name
    , name = (nsp + '/') + name;

  this.manager.onJoin(this.id, name);
  this.manager.store.publish('join', this.id, name);

  if (fn) {
    this.log.warn('Client#join callback is deprecated');
    fn();
  }

  return this;
};

/**
 * Un-joins a user from a room.
 *
 * @api public
 */

Socket.prototype.leave = function (name, fn) {
  var nsp = this.namespace.name
    , name = (nsp + '/') + name;

  this.manager.onLeave(this.id, name);
  this.manager.store.publish('leave', this.id, name);

  if (fn) {
    this.log.warn('Client#leave callback is deprecated');
    fn();
  }

  return this;
};

/**
 * Transmits a packet.
 *
 * @api private
 */

Socket.prototype.packet = function (packet) {
  if (this.flags.broadcast) {
    this.log.debug('broadcasting packet');
    this.namespace.in(this.flags.room).except(this.id).packet(packet);
  } else {
    packet.endpoint = this.flags.endpoint;
    packet = parser.encodePacket(packet);

    this.dispatch(packet, this.flags.volatile);
  }

  this.setFlags();

  return this;
};

/**
 * Dispatches a packet
 *
 * @api private
 */

Socket.prototype.dispatch = function (packet, volatile) {
  if (this.manager.transports[this.id] && this.manager.transports[this.id].open) {
    this.manager.transports[this.id].onDispatch(packet, volatile);
  } else {
    if (!volatile) {
      this.manager.onClientDispatch(this.id, packet, volatile);
    }

    this.manager.store.publish('dispatch-remote', this.id, packet, volatile);
  }
};

/**
 * Stores data for the client.
 *
 * @api public
 */

Socket.prototype.set = function (key, value, fn) {
  this.store.set(key, value, fn);
  return this;
};

/**
 * Retrieves data for the client
 *
 * @api public
 */

Socket.prototype.get = function (key, fn) {
  this.store.get(key, fn);
  return this;
};

/**
 * Checks data for the client
 *
 * @api public
 */

Socket.prototype.has = function (key, fn) {
  this.store.has(key, fn);
  return this;
};

/**
 * Deletes data for the client
 *
 * @api public
 */

Socket.prototype.del = function (key, fn) {
  this.store.del(key, fn);
  return this;
};

/**
 * Kicks client
 *
 * @api public
 */

Socket.prototype.disconnect = function () {
  if (!this.disconnected) {
    this.log.info('booting client');

    if ('' === this.namespace.name) {
      if (this.manager.transports[this.id] && this.manager.transports[this.id].open) {
        this.manager.transports[this.id].onForcedDisconnect();
      } else {
        this.manager.onClientDisconnect(this.id);
        this.manager.store.publish('disconnect-remote', this.id);
      }
    } else {
      this.packet({type: 'disconnect'});
      this.manager.onLeave(this.id, this.namespace.name);
      this.$emit('disconnect', 'booted');
    }

  }

  return this;
};

/**
 * Send a message.
 *
 * @api public
 */

Socket.prototype.send = function (data, fn) {
  var packet = {
      type: this.flags.json ? 'json' : 'message'
    , data: data
  };

  if (fn) {
    packet.id = ++this.ackPackets;
    packet.ack = true;
    this.acks[packet.id] = fn;
  }

  return this.packet(packet);
};

/**
 * Original emit function.
 *
 * @api private
 */

Socket.prototype.$emit = EventEmitter.prototype.emit;

/**
 * Emit override for custom events.
 *
 * @api public
 */

Socket.prototype.emit = function (ev) {
  if (ev == 'newListener') {
    return this.$emit.apply(this, arguments);
  }

  var args = util.toArray(arguments).slice(1)
    , lastArg = args[args.length - 1]
    , packet = {
          type: 'event'
        , name: ev
      };

  if ('function' == typeof lastArg) {
    packet.id = ++this.ackPackets;
    packet.ack = lastArg.length ? 'data' : true;
    this.acks[packet.id] = lastArg;
    args = args.slice(0, args.length - 1);
  }

  packet.args = args;

  return this.packet(packet);
};
