
/*!
* socket.io-node
* Copyright(c) 2011 LearnBoost <dev@learnboost.com>
* MIT Licensed
*/

/**
 * Module dependencies.
 */

var client = require('socket.io-client')
  , cp = require('child_process')
  , fs = require('fs')
  , util = require('./util');

/**
 * File type details.
 *
 * @api private
 */

var mime = {
    js: {
        type: 'application/javascript'
      , encoding: 'utf8'
      , gzip: true
    }
  , swf: {
        type: 'application/x-shockwave-flash'
      , encoding: 'binary'
      , gzip: false
    }
};

/**
 * Regexp for matching custom transport patterns. Users can configure their own
 * socket.io bundle based on the url structure. Different transport names are
 * concatinated using the `+` char. /socket.io/socket.io+websocket.js should
 * create a bundle that only contains support for the websocket.
 *
 * @api private
 */

var bundle = /\+((?:\+)?[\w\-]+)*(?:\.v\d+\.\d+\.\d+)?(?:\.js)$/
  , versioning = /\.v\d+\.\d+\.\d+(?:\.js)$/;

/**
 * Export the constructor
 */

exports = module.exports = Static;

/**
 * Static constructor
 *
 * @api public
 */

function Static (manager) {
  this.manager = manager;
  this.cache = {};
  this.paths = {};

  this.init();
}

/**
 * Initialize the Static by adding default file paths.
 *
 * @api public
 */

Static.prototype.init = function () {
  /**
   * Generates a unique id based the supplied transports array
   *
   * @param {Array} transports The array with transport types
   * @api private
   */
  function id (transports) {
    var id = transports.join('').split('').map(function (char) {
      return ('' + char.charCodeAt(0)).split('').pop();
    }).reduce(function (char, id) {
      return char +id;
    });

    return client.version + ':' + id;
  }

  /**
   * Generates a socket.io-client file based on the supplied transports.
   *
   * @param {Array} transports The array with transport types
   * @param {Function} callback Callback for the static.write
   * @api private
   */

  function build (transports, callback) {
    client.builder(transports, {
          minify: self.manager.enabled('browser client minification')
      }, function (err, content) {
        callback(err, content ? new Buffer(content) : null, id(transports));
      }
    );
  }

  var self = this;

  // add our default static files
  this.add('/static/flashsocket/WebSocketMain.swf', {
      file: client.dist + '/WebSocketMain.swf'
  });

  this.add('/static/flashsocket/WebSocketMainInsecure.swf', {
      file: client.dist + '/WebSocketMainInsecure.swf'
  });

  // generates dedicated build based on the available transports
  this.add('/socket.io.js', function (path, callback) {
    build(self.manager.get('transports'), callback);
  });

  this.add('/socket.io.v', { mime: mime.js }, function (path, callback) {
    build(self.manager.get('transports'), callback);
  });

  // allow custom builds based on url paths
  this.add('/socket.io+', { mime: mime.js }, function (path, callback) {
    var available = self.manager.get('transports')
      , matches = path.match(bundle)
      , transports = [];

    if (!matches) return callback('No valid transports');

    // make sure they valid transports
    matches[0].split('.')[0].split('+').slice(1).forEach(function (transport) {
      if (!!~available.indexOf(transport)) {
        transports.push(transport);
      }
    });

    if (!transports.length) return callback('No valid transports');
    build(transports, callback);
  });

  // clear cache when transports change
  this.manager.on('set:transports', function (key, value) {
    delete self.cache['/socket.io.js'];
    Object.keys(self.cache).forEach(function (key) {
      if (bundle.test(key)) {
        delete self.cache[key];
      }
    });
  });
};

/**
 * Gzip compress buffers.
 *
 * @param {Buffer} data The buffer that needs gzip compression
 * @param {Function} callback
 * @api public
 */

Static.prototype.gzip = function (data, callback) {
  var gzip = cp.spawn('gzip', ['-9', '-c', '-f', '-n'])
    , encoding = Buffer.isBuffer(data) ? 'binary' : 'utf8'
    , buffer = []
    , err;

  gzip.stdout.on('data', function (data) {
    buffer.push(data);
  });

  gzip.stderr.on('data', function (data) {
    err = data +'';
    buffer.length = 0;
  });

  gzip.on('close', function () {
    if (err) return callback(err);

    var size = 0
      , index = 0
      , i = buffer.length
      , content;

    while (i--) {
      size += buffer[i].length;
    }

    content = new Buffer(size);
    i = buffer.length;

    buffer.forEach(function (buffer) {
      var length = buffer.length;

      buffer.copy(content, index, 0, length);
      index += length;
    });

    buffer.length = 0;
    callback(null, content);
  });

  gzip.stdin.end(data, encoding);
};

/**
 * Is the path a static file?
 *
 * @param {String} path The path that needs to be checked
 * @api public
 */

Static.prototype.has = function (path) {
  // fast case
  if (this.paths[path]) return this.paths[path];

  var keys = Object.keys(this.paths)
    , i = keys.length;
 
  while (i--) {
    if (-~path.indexOf(keys[i])) return this.paths[keys[i]];
  }

  return false;
};

/**
 * Add new paths new paths that can be served using the static provider.
 *
 * @param {String} path The path to respond to
 * @param {Options} options Options for writing out the response
 * @param {Function} [callback] Optional callback if no options.file is
 * supplied this would be called instead.
 * @api public
 */

Static.prototype.add = function (path, options, callback) {
  var extension = /(?:\.(\w{1,4}))$/.exec(path);

  if (!callback && typeof options == 'function') {
    callback = options;
    options = {};
  }

  options.mime = options.mime || (extension ? mime[extension[1]] : false);

  if (callback) options.callback = callback;
  if (!(options.file || options.callback) || !options.mime) return false;

  this.paths[path] = options;

  return true;
};

/**
 * Writes a static response.
 *
 * @param {String} path The path for the static content
 * @param {HTTPRequest} req The request object
 * @param {HTTPResponse} res The response object
 * @api public
 */

Static.prototype.write = function (path, req, res) {
  /**
   * Write a response without throwing errors because can throw error if the
   * response is no longer writable etc.
   *
   * @api private
   */

  function write (status, headers, content, encoding) {
    try {
      res.writeHead(status, headers || undefined);

      // only write content if it's not a HEAD request and we actually have
      // some content to write (304's doesn't have content).
      res.end(
          req.method !== 'HEAD' && content ? content : ''
        , encoding || undefined
      );
    } catch (e) {}
  }

  /**
   * Answers requests depending on the request properties and the reply object.
   *
   * @param {Object} reply The details and content to reply the response with
   * @api private
   */

  function answer (reply) {
    var cached = req.headers['if-none-match'] === reply.etag;
    if (cached && self.manager.enabled('browser client etag')) {
      return write(304);
    }

    var accept = req.headers['accept-encoding'] || ''
      , gzip = !!~accept.toLowerCase().indexOf('gzip')
      , mime = reply.mime
      , versioned = reply.versioned
      , headers = {
          'Content-Type': mime.type
      };

    // check if we can add a etag
    if (self.manager.enabled('browser client etag') && reply.etag && !versioned) {
      headers['Etag'] = reply.etag;
    }

    // see if we need to set Expire headers because the path is versioned
    if (versioned) {
      var expires = self.manager.get('browser client expires');
      headers['Cache-Control'] = 'private, x-gzip-ok="", max-age=' + expires;
      headers['Date'] = new Date().toUTCString();
      headers['Expires'] = new Date(Date.now() + (expires * 1000)).toUTCString();
    }

    if (gzip && reply.gzip) {
      headers['Content-Length'] = reply.gzip.length;
      headers['Content-Encoding'] = 'gzip';
      headers['Vary'] = 'Accept-Encoding';
      write(200, headers, reply.gzip.content, mime.encoding);
    } else {
      headers['Content-Length'] = reply.length;
      write(200, headers, reply.content, mime.encoding);
    }

    self.manager.log.debug('served static content ' + path);
  }

  var self = this
    , details;

  // most common case first
  if (this.manager.enabled('browser client cache') && this.cache[path]) {
    return answer(this.cache[path]);
  } else if (this.manager.get('browser client handler')) {
    return this.manager.get('browser client handler').call(this, req, res);
  } else if ((details = this.has(path))) {
    /**
     * A small helper function that will let us deal with fs and dynamic files
     *
     * @param {Object} err Optional error
     * @param {Buffer} content The data
     * @api private
     */

    function ready (err, content, etag) {
      if (err) {
        self.manager.log.warn('Unable to serve file. ' + (err.message || err));
        return write(500, null, 'Error serving static ' + path);
      }

      // store the result in the cache
      var reply = self.cache[path] = {
            content: content
          , length: content.length
          , mime: details.mime
          , etag: etag || client.version
          , versioned: versioning.test(path)
        };

      // check if gzip is enabled
      if (details.mime.gzip && self.manager.enabled('browser client gzip')) {
        self.gzip(content, function (err, content) {
          if (!err) {
            reply.gzip = {
                content: content
              , length: content.length
            }
          }

          answer(reply);
        });
      } else {
        answer(reply);
      }
    }

    if (details.file) {
      fs.readFile(details.file, ready);
    } else if(details.callback) {
      details.callback.call(this, path, ready);
    } else {
      write(404, null, 'File handle not found');
    }
  } else {
    write(404, null, 'File not found');
  }
};
