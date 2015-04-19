/*
  node-http-proxy.js: Lookup table for proxy targets in node.js

  Copyright (c) 2010 Charlie Robbins

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
    events = require('events'),
    fs = require('fs'),
    url = require('url');

//
// ### function ProxyTable (router, silent)
// #### @router {Object} Object containing the host based routes
// #### @silent {Boolean} Value indicating whether we should suppress logs
// #### @hostnameOnly {Boolean} Value indicating if we should route based on __hostname string only__
// #### @pathnameOnly {Boolean} Value indicating if we should route based on only the pathname.  __This causes hostnames to be ignored.__.  Using this along with hostnameOnly wont work at all.
// Constructor function for the ProxyTable responsible for getting
// locations of proxy targets based on ServerRequest headers; specifically
// the HTTP host header.
//
var ProxyTable = exports.ProxyTable = function (options) {
  events.EventEmitter.call(this);

  this.silent       = options.silent || options.silent !== true;
  this.target       = options.target || {};
  this.pathnameOnly = options.pathnameOnly === true;
  this.hostnameOnly = options.hostnameOnly === true;

  if (typeof options.router === 'object') {
    //
    // If we are passed an object literal setup
    // the routes with RegExps from the router
    //
    this.setRoutes(options.router);
  }
  else if (typeof options.router === 'string') {
    //
    // If we are passed a string then assume it is a
    // file path, parse that file and watch it for changes
    //
    var self = this;
    this.routeFile = options.router;
    this.setRoutes(JSON.parse(fs.readFileSync(options.router)).router);

    fs.watchFile(this.routeFile, function () {
      fs.readFile(self.routeFile, function (err, data) {
        if (err) {
          self.emit('error', err);
        }

        self.setRoutes(JSON.parse(data).router);
        self.emit('routes', self.hostnameOnly === false ? self.routes : self.router);
      });
    });
  }
  else {
    throw new Error('Cannot parse router with unknown type: ' + typeof router);
  }
};

//
// Inherit from `events.EventEmitter`
//
util.inherits(ProxyTable, events.EventEmitter);

//
// ### function addRoute (route, target)
// #### @route {String} String containing route coming in
// #### @target {String} String containing the target
// Adds a host-based route to this instance.
//
ProxyTable.prototype.addRoute = function (route, target) {
  if (!this.router) {
    throw new Error('Cannot update ProxyTable routes without router.');
  }

  this.router[route] = target;
  this.setRoutes(this.router);
};

//
// ### function removeRoute (route)
// #### @route {String} String containing route to remove
// Removes a host-based route from this instance.
//
ProxyTable.prototype.removeRoute = function (route) {
  if (!this.router) {
    throw new Error('Cannot update ProxyTable routes without router.');
  }

  delete this.router[route];
  this.setRoutes(this.router);
};

//
// ### function setRoutes (router)
// #### @router {Object} Object containing the host based routes
// Sets the host-based routes to be used by this instance.
//
ProxyTable.prototype.setRoutes = function (router) {
  if (!router) {
    throw new Error('Cannot update ProxyTable routes without router.');
  }

  var self = this;
  this.router = router;

  if (this.hostnameOnly === false) {
    this.routes = [];

    Object.keys(router).forEach(function (path) {
      if (!/http[s]?/.test(router[path])) {
        router[path] = (self.target.https ? 'https://' : 'http://')
          + router[path];
      }

      var target = url.parse(router[path]),
          defaultPort = self.target.https ? 443 : 80;

      //
      // Setup a robust lookup table for the route:
      //
      //    {
      //      source: {
      //        regexp: /^foo.com/i,
      //        sref: 'foo.com',
      //        url: {
      //          protocol: 'http:',
      //          slashes: true,
      //          host: 'foo.com',
      //          hostname: 'foo.com',
      //          href: 'http://foo.com/',
      //          pathname: '/',
      //          path: '/'
      //        }
      //    },
      //    {
      //      target: {
      //        sref: '127.0.0.1:8000/',
      //        url: {
      //          protocol: 'http:',
      //          slashes: true,
      //          host: '127.0.0.1:8000',
      //          hostname: '127.0.0.1',
      //          href: 'http://127.0.0.1:8000/',
      //          pathname: '/',
      //          path: '/'
      //        }
      //    },
      //
      self.routes.push({
        source: {
          regexp: new RegExp('^' + path, 'i'),
          sref: path,
          url: url.parse('http://' + path)
        },
        target: {
          sref: target.hostname + ':' + (target.port || defaultPort) + target.path,
          url: target
        }
      });
    });
  }
};

//
// ### function getProxyLocation (req)
// #### @req {ServerRequest} The incoming server request to get proxy information about.
// Returns the proxy location based on the HTTP Headers in the  ServerRequest `req`
// available to this instance.
//
ProxyTable.prototype.getProxyLocation = function (req) {
  if (!req || !req.headers || !req.headers.host) {
    return null;
  }

  var targetHost = req.headers.host.split(':')[0];
  if (this.hostnameOnly === true) {
    var target = targetHost;
    if (this.router.hasOwnProperty(target)) {
      var location = this.router[target].split(':'),
          host = location[0],
          port = location.length === 1 ? 80 : location[1];

      return {
        port: port,
        host: host
      };
    }
  }
  else if (this.pathnameOnly === true) {
    var target = req.url;
    for (var i in this.routes) {
      var route = this.routes[i];
      //
      // If we are matching pathname only, we remove the matched pattern.
      //
      // IE /wiki/heartbeat
      // is redirected to
      // /heartbeat
      //
      // for the route "/wiki" : "127.0.0.1:8020"
      //
      if (target.match(route.source.regexp)) {
        req.url = url.format(target.replace(route.source.regexp, ''));
        return {
          protocol: route.target.url.protocol.replace(':', ''),
          host: route.target.url.hostname,
          port: route.target.url.port
            || (this.target.https ? 443 : 80)
        };
      }
    }

  }
  else {
    var target = targetHost + req.url;
    for (var i in this.routes) {
      var route = this.routes[i];
      if (target.match(route.source.regexp)) {
        //
        // Attempt to perform any path replacement for differences
        // between the source path and the target path. This replaces the
        // path's part of the URL to the target's part of the URL.
        //
        // 1. Parse the request URL
        // 2. Replace any portions of the source path with the target path
        // 3. Set the request URL to the formatted URL with replacements.
        //
        var parsed = url.parse(req.url);

        parsed.pathname = parsed.pathname.replace(
          route.source.url.pathname,
          route.target.url.pathname
        );

        req.url = url.format(parsed);

        return {
          protocol: route.target.url.protocol.replace(':', ''),
          host: route.target.url.hostname,
          port: route.target.url.port
            || (this.target.https ? 443 : 80)
        };
      }
    }
  }

  return null;
};

//
// ### close function ()
// Cleans up the event listeneners maintained
// by this instance.
//
ProxyTable.prototype.close = function () {
  if (typeof this.routeFile === 'string') {
    fs.unwatchFile(this.routeFile);
  }
};
