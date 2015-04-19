var bytes = require('bytes')
var iconv = require('iconv-lite')

module.exports = function (stream, options, done) {
  if (options === true || typeof options === 'string') {
    // short cut for encoding
    options = {
      encoding: options
    }
  }

  options = options || {}

  if (typeof options === 'function') {
    done = options
    options = {}
  }

  // get encoding
  var encoding = options.encoding !== true
    ? options.encoding
    : 'utf-8'

  // convert the limit to an integer
  var limit = null
  if (typeof options.limit === 'number')
    limit = options.limit
  if (typeof options.limit === 'string')
    limit = bytes(options.limit)

  // convert the expected length to an integer
  var length = null
  if (options.length != null && !isNaN(options.length))
    length = parseInt(options.length, 10)

  // check the length and limit options.
  // note: we intentionally leave the stream paused,
  // so users should handle the stream themselves.
  if (limit !== null && length !== null && length > limit) {
    var err = makeError('request entity too large', 'entity.too.large')
    err.status = err.statusCode = 413
    err.length = err.expected = length
    err.limit = limit
    cleanup()
    halt(stream)
    process.nextTick(function () {
      done(err)
    })
    return defer
  }

  // streams1: assert request encoding is buffer.
  // streams2+: assert the stream encoding is buffer.
  //   stream._decoder: streams1
  //   state.encoding: streams2
  //   state.decoder: streams2, specifically < 0.10.6
  var state = stream._readableState
  if (stream._decoder || (state && (state.encoding || state.decoder))) {
    // developer error
    var err = makeError('stream encoding should not be set',
      'stream.encoding.set')
    err.status = err.statusCode = 500
    cleanup()
    halt(stream)
    process.nextTick(function () {
      done(err)
    })
    return defer
  }

  var received = 0
  var decoder

  try {
    decoder = getDecoder(encoding)
  } catch (err) {
    cleanup()
    halt(stream)
    process.nextTick(function () {
      done(err)
    })
    return defer
  }

  var buffer = decoder
    ? ''
    : []

  stream.on('data', onData)
  stream.once('end', onEnd)
  stream.once('error', onEnd)
  stream.once('close', cleanup)

  return defer

  // yieldable support
  function defer(fn) {
    done = fn
  }

  function onData(chunk) {
    received += chunk.length
    decoder
      ? buffer += decoder.write(chunk)
      : buffer.push(chunk)

    if (limit !== null && received > limit) {
      var err = makeError('request entity too large', 'entity.too.large')
      err.status = err.statusCode = 413
      err.received = received
      err.limit = limit
      cleanup()
      halt(stream)
      done(err)
    }
  }

  function onEnd(err) {
    if (err) {
      cleanup()
      halt(stream)
      done(err)
    } else if (length !== null && received !== length) {
      err = makeError('request size did not match content length',
        'request.size.invalid')
      err.status = err.statusCode = 400
      err.received = received
      err.length = err.expected = length
      cleanup()
      done(err)
    } else {
      var string = decoder
        ? buffer + (decoder.end() || '')
        : Buffer.concat(buffer)
      cleanup()
      done(null, string)
    }
  }

  function cleanup() {
    received = buffer = null

    stream.removeListener('data', onData)
    stream.removeListener('end', onEnd)
    stream.removeListener('error', onEnd)
    stream.removeListener('close', cleanup)
  }
}

function getDecoder(encoding) {
  if (!encoding) return null

  try {
    return iconv.getCodec(encoding).decoder()
  } catch (e) {
    var err = makeError('specified encoding unsupported', 'encoding.unsupported')
    err.status = err.statusCode = 415
    err.encoding = encoding
    throw err
  }
}

/**
 * Halt a stream.
 *
 * @param {Object} stream
 * @api private
 */

function halt(stream) {
  // unpipe everything from the stream
  unpipe(stream)

  // pause stream
  if (typeof stream.pause === 'function') {
    stream.pause()
  }
}

// to create serializable errors you must re-set message so
// that it is enumerable and you must re configure the type
// property so that is writable and enumerable
function makeError(message, type) {
  var error = new Error()
  error.message = message
  Object.defineProperty(error, 'type', {
    value: type,
    enumerable: true,
    writable: true,
    configurable: true
  })
  return error
}

/**
 * Unpipe everything from a stream.
 *
 * @param {Object} stream
 * @api private
 */

/* istanbul ignore next: implementation differs between versions */
function unpipe(stream) {
  if (typeof stream.unpipe === 'function') {
    // new-style
    stream.unpipe()
    return
  }

  // Node.js 0.8 hack
  var listener
  var listeners = stream.listeners('close')

  for (var i = 0; i < listeners.length; i++) {
    listener = listeners[i]

    if (listener.name !== 'cleanup' && listener.name !== 'onclose') {
      continue
    }

    // invoke the listener
    listener.call(stream)
  }
}
