/*
 * response-stream.js: A Stream focused on writing any relevant information to
 * a raw http.ServerResponse object.
 *
 * (C) 2011, Nodejitsu Inc.
 * MIT LICENSE
 *
 */

var util = require('util'),
    HttpStream = require('./http-stream');

//
// ### function ResponseStream (options)
//
//
var ResponseStream = module.exports = function (options) {
  var self = this,
      key;

  options = options || {};
  HttpStream.call(this, options);

  this.writeable = true;
  this.response = options.response;

  if (options.headers) {
    for (key in options.headers) {
      this.response.setHeader(key, options.headers[key]);
    }
  }

  //
  // Proxy `statusCode` changes to the actual `response.statusCode`.
  //
  Object.defineProperty(this, 'statusCode', {
    get: function () {
      return self.response.statusCode;
    },
    set: function (value) {
      self.response.statusCode = value;
    },
    enumerable: true,
    configurable: true
  });

  if (this.response) {
    this._headers = this.response._headers = this.response._headers || {};

    // Patch to node core
    this.response._headerNames = this.response._headerNames || {};

    //
    // Proxy to emit "header" event
    //
    this._renderHeaders = this.response._renderHeaders;
    this.response._renderHeaders = function () {
      if (!self._emittedHeader) {
        self._emittedHeader = true;
        self.headerSent = true;
        self.emit('header');
      }
      return self._renderHeaders.call(self.response);
    };
  }
};

util.inherits(ResponseStream, HttpStream);

ResponseStream.prototype.writeHead = function (statusCode, headers) {
  if (headers) {
    for (var key in headers) {
      this.response.setHeader(key, headers[key]);
    }
  }

  this.response.statusCode = statusCode;
};

//
// Create pass-thru for the necessary
// `http.ServerResponse` methods.
//
['setHeader', 'getHeader', 'removeHeader', '_implicitHeader', 'addTrailers'].forEach(function (method) {
  ResponseStream.prototype[method] = function () {
    return this.response[method].apply(this.response, arguments);
  };
});

ResponseStream.prototype.json = function (obj) {
  if (!this.response.writable) {
    return;
  }

  if (typeof obj === 'number') {
    this.response.statusCode = obj;
    obj = arguments[1];
  }

  this.modified = true;

  if (!this.response._header && this.response.getHeader('content-type') !== 'application/json') {
    this.response.setHeader('content-type', 'application/json');
  }

  this.end(obj ? JSON.stringify(obj) : '');
};

ResponseStream.prototype.html = function (str) {
  if (!this.response.writable) {
    return;
  }

  if (typeof str === 'number') {
    this.response.statusCode = str;
    str = arguments[1];
  }

  this.modified = true;

  if (!this.response._header && this.response.getHeader('content-type') !== 'text/html') {
    this.response.setHeader('content-type', 'text/html');
  }

  this.end(str ? str: '');
};

ResponseStream.prototype.text = function (str) {
  if (!this.response.writable) {
    return;
  }

  if (typeof str === 'number') {
    this.response.statusCode = str;
    str = arguments[1];
  }

  this.modified = true;

  if (!this.response._header && this.response.getHeader('content-type') !== 'text/plain') {
    this.response.setHeader('content-type', 'text/plain');
  }

  this.end(str ? str: '');
};

ResponseStream.prototype.end = function (data) {
  if (data && this.writable) {
    this.emit('data', data);
  }

  this.modified = true;
  this.emit('end');
};

ResponseStream.prototype.write = function (data) {
  this.modified = true;

  if (this.writable) {
    this.emit('data', data);
  }
};

ResponseStream.prototype.redirect = function (path, status) {
  var url = '';

  if (~path.indexOf('://')) {
    url = path;
  } else {
    url += this.req.connection.encrypted ? 'https://' : 'http://';
    url += this.req.headers.host;
    url += (path[0] === '/') ? path : '/' + path;
  }

  this.res.writeHead(status || 302, { 'Location': url });
  this.end();
};
