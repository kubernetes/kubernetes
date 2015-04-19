/*
  node-http-proxy.js: http proxy for node.js

  Copyright (c) 2010 Charlie Robbins, Mikeal Rogers, Marak Squires, Fedor Indutny

  Permission is hereby granted, free of charge, to any person obtaining
  a copy of this software and associated documentation files (the
  "Software"), to deal in the Software without restriction, including
  without limitation the rights to use, copy, modify, merge, publish,
  distribute, sublicense, and/or sell copies of the Software, and to
  permit persons to whom the Software is furnished to do so, subject to
  the following conditions:

  The above copyright notice and this permission notice shall be
  included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/

var util = require('util'),
    http = require('http'),
    https = require('https'),
    events = require('events'),
    maxSockets = 100;

//
// Expose version information through `pkginfo`.
//
require('pkginfo')(module, 'version');

//
// ### Export the relevant objects exposed by `node-http-proxy`
//
var HttpProxy    = exports.HttpProxy    = require('./node-http-proxy/http-proxy').HttpProxy,
    ProxyTable   = exports.ProxyTable   = require('./node-http-proxy/proxy-table').ProxyTable,
    RoutingProxy = exports.RoutingProxy = require('./node-http-proxy/routing-proxy').RoutingProxy;

//
// ### function createServer ([port, host, options, handler])
// #### @port {number} **Optional** Port to use on the proxy target host.
// #### @host {string} **Optional** Host of the proxy target.
// #### @options {Object} **Optional** Options for the HttpProxy instance used
// #### @handler {function} **Optional** Request handler for the server
// Returns a server that manages an instance of HttpProxy. Flexible arguments allow for:
//
// * `httpProxy.createServer(9000, 'localhost')`
// * `httpProxy.createServer(9000, 'localhost', options)
// * `httpPRoxy.createServer(function (req, res, proxy) { ... })`
//
exports.createServer = function () {
  var args = Array.prototype.slice.call(arguments),
      handlers = [],
      callback,
      options = {},
      message,
      handler,
      server,
      proxy,
      host,
      port;

  //
  // Liberally parse arguments of the form:
  //
  //    httpProxy.createServer('localhost', 9000, callback);
  //    httpProxy.createServer({ host: 'localhost', port: 9000 }, callback);
  //    **NEED MORE HERE!!!**
  //
  args.forEach(function (arg) {
    arg = Number(arg) || arg;
    switch (typeof arg) {
      case 'string':   host = arg; break;
      case 'number':   port = arg; break;
      case 'object':   options = arg || {}; break;
      case 'function': callback = arg; handlers.push(callback); break;
    };
  });

  //
  // Helper function to create intelligent error message(s)
  // for the very liberal arguments parsing performed by
  // `require('http-proxy').createServer()`.
  //
  function validArguments() {
    var conditions = {
      'port and host': function () {
        return port && host;
      },
      'options.target or options.router': function () {
        return options && (options.router ||
          (options.target && options.target.host && options.target.port));
      },
      'or proxy handlers': function () {
        return handlers && handlers.length;
      }
    }

    var missing = Object.keys(conditions).filter(function (name) {
      return !conditions[name]();
    });

    if (missing.length === 3) {
      message = 'Cannot proxy without ' + missing.join(', ');
      return false;
    }

    return true;
  }

  if (!validArguments()) {
    //
    // If `host`, `port` and `options` are all not passed (with valid
    // options) then this server is improperly configured.
    //
    throw new Error(message);
    return;
  }

  //
  // Hoist up any explicit `host` or `port` arguments
  // that have been passed in to the options we will
  // pass to the `httpProxy.HttpProxy` constructor.
  //
  options.target      = options.target      || {};
  options.target.port = options.target.port || port;
  options.target.host = options.target.host || host;

  if (options.target && options.target.host && options.target.port) {
    //
    // If an explicit `host` and `port` combination has been passed
    // to `.createServer()` then instantiate a hot-path optimized
    // `HttpProxy` object and add the "proxy" middleware layer.
    //
    proxy = new HttpProxy(options);
    handlers.push(function (req, res) {
      proxy.proxyRequest(req, res);
    });
  }
  else {
    //
    // If no explicit `host` or `port` combination has been passed then
    // we have to assume that this is a "go-anywhere" Proxy (i.e. a `RoutingProxy`).
    //
    proxy = new RoutingProxy(options);

    if (options.router) {
      //
      // If a routing table has been supplied than we assume
      // the user intends us to add the "proxy" middleware layer
      // for them
      //
      handlers.push(function (req, res) {
        proxy.proxyRequest(req, res);
      });

      proxy.on('routes', function (routes) {
        server.emit('routes', routes);
      });
    }
  }

  //
  // Create the `http[s].Server` instance which will use
  // an instance of `httpProxy.HttpProxy`.
  //
  handler = handlers.length > 1
    ? exports.stack(handlers, proxy)
    : function (req, res) { handlers[0](req, res, proxy) };

  server  = options.https
    ? https.createServer(options.https, handler)
    : http.createServer(handler);

  server.on('close', function () {
    proxy.close();
  });

  if (!callback) {
    //
    // If an explicit callback has not been supplied then
    // automagically proxy the request using the `HttpProxy`
    // instance we have created.
    //
    server.on('upgrade', function (req, socket, head) {
      proxy.proxyWebSocketRequest(req, socket, head);
    });
  }

  //
  // Set the proxy on the server so it is available
  // to the consumer of the server
  //
  server.proxy = proxy;
  return server;
};

//
// ### function buffer (obj)
// #### @obj {Object} Object to pause events from
// Buffer `data` and `end` events from the given `obj`.
// Consumers of HttpProxy performing async tasks
// __must__ utilize this utility, to re-emit data once
// the async operation has completed, otherwise these
// __events will be lost.__
//
//      var buffer = httpProxy.buffer(req);
//      fs.readFile(path, function () {
//         httpProxy.proxyRequest(req, res, host, port, buffer);
//      });
//
// __Attribution:__ This approach is based heavily on
// [Connect](https://github.com/senchalabs/connect/blob/master/lib/utils.js#L157).
// However, this is not a big leap from the implementation in node-http-proxy < 0.4.0.
// This simply chooses to manage the scope of the events on a new Object literal as opposed to
// [on the HttpProxy instance](https://github.com/nodejitsu/node-http-proxy/blob/v0.3.1/lib/node-http-proxy.js#L154).
//
exports.buffer = function (obj) {
  var events = [],
      onData,
      onEnd;

  obj.on('data', onData = function (data, encoding) {
    events.push(['data', data, encoding]);
  });

  obj.on('end', onEnd = function (data, encoding) {
    events.push(['end', data, encoding]);
  });

  return {
    end: function () {
      obj.removeListener('data', onData);
      obj.removeListener('end', onEnd);
    },
    destroy: function () {
      this.end();
     	this.resume = function () {
     	  console.error("Cannot resume buffer after destroying it.");
     	};

     	onData = onEnd = events = obj = null;
    },
    resume: function () {
      this.end();
      for (var i = 0, len = events.length; i < len; ++i) {
        obj.emit.apply(obj, events[i]);
      }
    }
  };
};

//
// ### function getMaxSockets ()
// Returns the maximum number of sockets
// allowed on __every__ outgoing request
// made by __all__ instances of `HttpProxy`
//
exports.getMaxSockets = function () {
  return maxSockets;
};

//
// ### function setMaxSockets ()
// Sets the maximum number of sockets
// allowed on __every__ outgoing request
// made by __all__ instances of `HttpProxy`
//
exports.setMaxSockets = function (value) {
  maxSockets = value;
};

//
// ### function stack (middlewares, proxy)
// #### @middlewares {Array} Array of functions to stack.
// #### @proxy {HttpProxy|RoutingProxy} Proxy instance to
// Iteratively build up a single handler to the `http.Server`
// `request` event (i.e. `function (req, res)`) by wrapping
// each middleware `layer` into a `child` middleware which
// is in invoked by the parent (i.e. predecessor in the Array).
//
// adapted from https://github.com/creationix/stack
//
exports.stack = function stack (middlewares, proxy) {
  var handle;
  middlewares.reverse().forEach(function (layer) {
    var child = handle;
    handle = function (req, res) {
      var next = function (err) {
        if (err) {
          if (! proxy.emit('middlewareError', err, req, res)) {
            console.error('Error in middleware(s): %s', err.stack);
          }

          if (res._headerSent) {
            res.destroy();
          }
          else {
            res.statusCode = 500;
            res.setHeader('Content-Type', 'text/plain');
            res.end('Internal Server Error');
          }

          return;
        }

        if (child) {
          child(req, res);
        }
      };

      //
      // Set the prototype of the `next` function to the instance
      // of the `proxy` so that in can be used interchangably from
      // a `connect` style callback and a true `HttpProxy` object.
      //
      // e.g. `function (req, res, next)` vs. `function (req, res, proxy)`
      //
      next.__proto__ = proxy;
      layer(req, res, next);
    };
  });

  return handle;
};

//
// ### function _getAgent (host, port, secure)
// #### @options {Object} Options to use when creating the agent.
//
//    {
//      host: 'localhost',
//      port: 9000,
//      https: true,
//      maxSockets: 100
//    }
//
// Createsan agent from the `http` or `https` module
// and sets the `maxSockets` property appropriately.
//
exports._getAgent = function _getAgent (options) {
  if (!options || !options.host) {
    throw new Error('`options.host` is required to create an Agent.');
  }

  if (!options.port) {
    options.port = options.https ? 443 : 80;
  }

  var Agent = options.https ? https.Agent : http.Agent,
      agent;

  // require('http-proxy').setMaxSockets() should override http's default
  // configuration value (which is pretty low).
  options.maxSockets = options.maxSockets || maxSockets;
  agent = new Agent(options);

  return agent;
}

//
// ### function _getProtocol (options)
// #### @options {Object} Options for the proxy target.
// Returns the appropriate node.js core protocol module (i.e. `http` or `https`)
// based on the `options` supplied.
//
exports._getProtocol = function _getProtocol (options) {
  return options.https ? https : http;
};


//
// ### function _getBase (options)
// #### @options {Object} Options for the proxy target.
// Returns the relevate base object to create on outgoing proxy request.
// If `options.https` are supplied, this function respond with an object
// containing the relevant `ca`, `key`, and `cert` properties.
//
exports._getBase = function _getBase (options) {
  var result = function () {};

  if (options.https && typeof options.https === 'object') {
    ['ca', 'cert', 'key'].forEach(function (key) {
      if (options.https[key]) {
        result.prototype[key] = options.https[key];
      }
    });
  }

  return result;
};
