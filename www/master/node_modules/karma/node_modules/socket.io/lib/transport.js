
/*!
 * socket.io-node
 * Copyright(c) 2011 LearnBoost <dev@learnboost.com>
 * MIT Licensed
 */

/**
 * Module dependencies.
 */

var parser = require('./parser');

/**
 * Expose the constructor.
 */

exports = module.exports = Transport;

/**
 * Transport constructor.
 *
 * @api public
 */

function Transport (mng, data, req) {
  this.manager = mng;
  this.id = data.id;
  this.disconnected = false;
  this.drained = true;
  this.handleRequest(req);
};

/**
 * Access the logger.
 *
 * @api public
 */

Transport.prototype.__defineGetter__('log', function () {
  return this.manager.log;
});

/**
 * Access the store.
 *
 * @api public
 */

Transport.prototype.__defineGetter__('store', function () {
  return this.manager.store;
});

/**
 * Handles a request when it's set.
 *
 * @api private
 */

Transport.prototype.handleRequest = function (req) {
  this.log.debug('setting request', req.method, req.url);
  this.req = req;

  if (req.method == 'GET') {
    this.socket = req.socket;
    this.open = true;
    this.drained = true;
    this.setHeartbeatInterval();

    this.setHandlers();
    this.onSocketConnect();
  }
};

/**
 * Called when a connection is first set.
 *
 * @api private
 */

Transport.prototype.onSocketConnect = function () { };

/**
 * Sets transport handlers
 *
 * @api private
 */

Transport.prototype.setHandlers = function () {
  var self = this;

  this.bound = {
      end: this.onSocketEnd.bind(this)
    , close: this.onSocketClose.bind(this)
    , error: this.onSocketError.bind(this)
    , drain: this.onSocketDrain.bind(this)
  };

  this.socket.on('end', this.bound.end);
  this.socket.on('close', this.bound.close);
  this.socket.on('error', this.bound.error);
  this.socket.on('drain', this.bound.drain);

  this.handlersSet = true;
};

/**
 * Removes transport handlers
 *
 * @api private
 */

Transport.prototype.clearHandlers = function () {
  if (this.handlersSet) {
    this.socket.removeListener('end', this.bound.end);
    this.socket.removeListener('close', this.bound.close);
    this.socket.removeListener('error', this.bound.error);
    this.socket.removeListener('drain', this.bound.drain);
  }
};

/**
 * Called when the connection dies
 *
 * @api private
 */

Transport.prototype.onSocketEnd = function () {
  this.end('socket end');
};

/**
 * Called when the connection dies
 *
 * @api private
 */

Transport.prototype.onSocketClose = function (error) {
  this.end(error ? 'socket error' : 'socket close');
};

/**
 * Called when the connection has an error.
 *
 * @api private
 */

Transport.prototype.onSocketError = function (err) {
  if (this.open) {
    this.socket.destroy();
    this.onClose();
  }

  this.log.info('socket error '  + err.stack);
};

/**
 * Called when the connection is drained.
 *
 * @api private
 */

Transport.prototype.onSocketDrain = function () {
  this.drained = true;
};

/**
 * Called upon receiving a heartbeat packet.
 *
 * @api private
 */

Transport.prototype.onHeartbeatClear = function () {
  this.clearHeartbeatTimeout();
  this.setHeartbeatInterval();
};

/**
 * Called upon a forced disconnection.
 *
 * @api private
 */

Transport.prototype.onForcedDisconnect = function () {
  if (!this.disconnected) {
    this.log.info('transport end by forced client disconnection');
    if (this.open) {
      this.packet({ type: 'disconnect' });
    }
    this.end('booted');
  }
};

/**
 * Dispatches a packet.
 *
 * @api private
 */

Transport.prototype.onDispatch = function (packet, volatile) {
  if (volatile) {
    this.writeVolatile(packet);
  } else {
    this.write(packet);
  }
};

/**
 * Sets the close timeout.
 */

Transport.prototype.setCloseTimeout = function () {
  if (!this.closeTimeout) {
    var self = this;

    this.closeTimeout = setTimeout(function () {
      self.log.debug('fired close timeout for client', self.id);
      self.closeTimeout = null;
      self.end('close timeout');
    }, this.manager.get('close timeout') * 1000);

    this.log.debug('set close timeout for client', this.id);
  }
};

/**
 * Clears the close timeout.
 */

Transport.prototype.clearCloseTimeout = function () {
  if (this.closeTimeout) {
    clearTimeout(this.closeTimeout);
    this.closeTimeout = null;

    this.log.debug('cleared close timeout for client', this.id);
  }
};

/**
 * Sets the heartbeat timeout
 */

Transport.prototype.setHeartbeatTimeout = function () {
  if (!this.heartbeatTimeout && this.manager.enabled('heartbeats')) {
    var self = this;

    this.heartbeatTimeout = setTimeout(function () {
      self.log.debug('fired heartbeat timeout for client', self.id);
      self.heartbeatTimeout = null;
      self.end('heartbeat timeout');
    }, this.manager.get('heartbeat timeout') * 1000);

    this.log.debug('set heartbeat timeout for client', this.id);
  }
};

/**
 * Clears the heartbeat timeout
 *
 * @param text
 */

Transport.prototype.clearHeartbeatTimeout = function () {
  if (this.heartbeatTimeout && this.manager.enabled('heartbeats')) {
    clearTimeout(this.heartbeatTimeout);
    this.heartbeatTimeout = null;
    this.log.debug('cleared heartbeat timeout for client', this.id);
  }
};

/**
 * Sets the heartbeat interval. To be called when a connection opens and when
 * a heartbeat is received.
 *
 * @api private
 */

Transport.prototype.setHeartbeatInterval = function () {
  if (!this.heartbeatInterval && this.manager.enabled('heartbeats')) {
    var self = this;

    this.heartbeatInterval = setTimeout(function () {
      self.heartbeat();
      self.heartbeatInterval = null;
    }, this.manager.get('heartbeat interval') * 1000);

    this.log.debug('set heartbeat interval for client', this.id);
  }
};

/**
 * Clears all timeouts.
 *
 * @api private
 */

Transport.prototype.clearTimeouts = function () {
  this.clearCloseTimeout();
  this.clearHeartbeatTimeout();
  this.clearHeartbeatInterval();
};

/**
 * Sends a heartbeat
 *
 * @api private
 */

Transport.prototype.heartbeat = function () {
  if (this.open) {
    this.log.debug('emitting heartbeat for client', this.id);
    this.packet({ type: 'heartbeat' });
    this.setHeartbeatTimeout();
  }

  return this;
};

/**
 * Handles a message.
 *
 * @param {Object} packet object
 * @api private
 */

Transport.prototype.onMessage = function (packet) {
  var current = this.manager.transports[this.id];

  if ('heartbeat' == packet.type) {
    this.log.debug('got heartbeat packet');

    if (current && current.open) {
      current.onHeartbeatClear();
    } else {
      this.store.publish('heartbeat-clear', this.id);
    }
  } else {
    if ('disconnect' == packet.type && packet.endpoint == '') {
      this.log.debug('got disconnection packet');

      if (current) {
        current.onForcedDisconnect();
      } else {
        this.store.publish('disconnect-force', this.id);
      }

      return;
    }

    if (packet.id && packet.ack != 'data') {
      this.log.debug('acknowledging packet automatically');

      var ack = parser.encodePacket({
          type: 'ack'
        , ackId: packet.id
        , endpoint: packet.endpoint || ''
      });

      if (current && current.open) {
        current.onDispatch(ack);
      } else {
        this.manager.onClientDispatch(this.id, ack);
        this.store.publish('dispatch-remote', this.id, ack);
      }
    }

    // handle packet locally or publish it
    if (current) {
      this.manager.onClientMessage(this.id, packet);
    } else {
      this.store.publish('message-remote', this.id, packet);
    }
  }
};

/**
 * Clears the heartbeat interval
 *
 * @api private
 */

Transport.prototype.clearHeartbeatInterval = function () {
  if (this.heartbeatInterval && this.manager.enabled('heartbeats')) {
    clearTimeout(this.heartbeatInterval);
    this.heartbeatInterval = null;
    this.log.debug('cleared heartbeat interval for client', this.id);
  }
};

/**
 * Finishes the connection and makes sure client doesn't reopen
 *
 * @api private
 */

Transport.prototype.disconnect = function (reason) {
  this.packet({ type: 'disconnect' });
  this.end(reason);

  return this;
};

/**
 * Closes the connection.
 *
 * @api private
 */

Transport.prototype.close = function () {
  if (this.open) {
    this.doClose();
    this.onClose();
  }
};

/**
 * Called upon a connection close.
 *
 * @api private
 */

Transport.prototype.onClose = function () {
  if (this.open) {
    this.setCloseTimeout();
    this.clearHandlers();
    this.open = false;
    this.manager.onClose(this.id);
    this.store.publish('close', this.id);
  }
};

/**
 * Cleans up the connection, considers the client disconnected.
 *
 * @api private
 */

Transport.prototype.end = function (reason) {
  if (!this.disconnected) {
    this.log.info('transport end (' + reason + ')');

    var local = this.manager.transports[this.id];

    this.close();
    this.clearTimeouts();
    this.disconnected = true;

    if (local) {
      this.manager.onClientDisconnect(this.id, reason);
    }

    this.store.publish('disconnect-remote', this.id, reason);
  }
};

/**
 * Signals that the transport should pause and buffer data.
 *
 * @api public
 */

Transport.prototype.discard = function () {
  this.log.debug('discarding transport');
  this.discarded = true;
  this.clearTimeouts();
  this.clearHandlers();

  return this;
};

/**
 * Writes an error packet with the specified reason and advice.
 *
 * @param {Number} advice
 * @param {Number} reason
 * @api public
 */

Transport.prototype.error = function (reason, advice) {
  this.packet({
      type: 'error'
    , reason: reason
    , advice: advice
  });

  this.log.warn(reason, advice ? ('client should ' + advice) : '');
  this.end('error');
};

/**
 * Write a packet.
 *
 * @api public
 */

Transport.prototype.packet = function (obj) {
  return this.write(parser.encodePacket(obj));
};

/**
 * Writes a volatile message.
 *
 * @api private
 */

Transport.prototype.writeVolatile = function (msg) {
  if (this.open) {
    if (this.drained) {
      this.write(msg);
    } else {
      this.log.debug('ignoring volatile packet, buffer not drained');
    }
  } else {
    this.log.debug('ignoring volatile packet, transport not open');
  }
};
