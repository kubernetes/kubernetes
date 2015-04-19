/*!
 * on-finished
 * Copyright(c) 2013 Jonathan Ong
 * Copyright(c) 2014 Douglas Christopher Wilson
 * MIT Licensed
 */

/**
 * Module exports.
 */

module.exports = onFinished;
module.exports.isFinished = isFinished;

/**
* Module dependencies.
*/

var first = require('ee-first')

/**
* Variables.
*/

/* istanbul ignore next */
var defer = typeof setImmediate === 'function'
  ? setImmediate
  : function(fn){ process.nextTick(fn.bind.apply(fn, arguments)) }

/**
 * Invoke callback when the response has finished, useful for
 * cleaning up resources afterwards.
 *
 * @param {object} msg
 * @param {function} listener
 * @return {object}
 * @api public
 */

function onFinished(msg, listener) {
  if (isFinished(msg) !== false) {
    defer(listener)
    return msg
  }

  // attach the listener to the message
  attachListener(msg, listener)

  return msg
}

/**
 * Determine is message is already finished.
 *
 * @param {object} msg
 * @return {boolean}
 * @api public
 */

function isFinished(msg) {
  var socket = msg.socket

  if (typeof msg.finished === 'boolean') {
    // OutgoingMessage
    return Boolean(!socket || msg.finished || !socket.writable)
  }

  if (typeof msg.complete === 'boolean') {
    // IncomingMessage
    return Boolean(!socket || msg.complete || !socket.readable)
  }

  // don't know
  return undefined
}

/**
 * Attach the listener to the message.
 *
 * @param {object} msg
 * @return {function}
 * @api private
 */

function attachListener(msg, listener) {
  var attached = msg.__onFinished
  var socket = msg.socket

  // create a private single listener with queue
  if (!attached || !attached.queue) {
    attached = msg.__onFinished = createListener(msg)

    // finished on first event
    first([
      [socket, 'error', 'close'],
      [msg, 'end', 'finish'],
    ], attached)
  }

  attached.queue.push(listener)
}

/**
 * Create listener on message.
 *
 * @param {object} msg
 * @return {function}
 * @api private
 */

function createListener(msg) {
  function listener(err) {
    if (msg.__onFinished === listener) msg.__onFinished = null
    if (!listener.queue) return

    var queue = listener.queue
    listener.queue = null

    for (var i = 0; i < queue.length; i++) {
      queue[i](err)
    }
  }

  listener.queue = []

  return listener
}
