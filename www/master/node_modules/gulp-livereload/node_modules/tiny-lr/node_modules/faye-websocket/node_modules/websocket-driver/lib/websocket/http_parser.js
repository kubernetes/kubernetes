'use strict';

var NodeHTTPParser = process.binding('http_parser').HTTPParser,
    version        = NodeHTTPParser.RESPONSE ? 6 : 4;

var HttpParser = function(type) {
  if (type === 'request')
    this._parser = new NodeHTTPParser(NodeHTTPParser.REQUEST || 'request');
  else
    this._parser = new NodeHTTPParser(NodeHTTPParser.RESPONSE || 'response');

  this._type     = type;
  this._complete = false;
  this.headers   = {};

  var current = null,
      self    = this;

  this._parser.onHeaderField = function(b, start, length) {
    current = b.toString('utf8', start, start + length).toLowerCase();
  };

  this._parser.onHeaderValue = function(b, start, length) {
    var value = b.toString('utf8', start, start + length);

    if (self.headers.hasOwnProperty(current))
      self.headers[current] += ', ' + value;
    else
      self.headers[current] = value;
  };

  this._parser.onHeadersComplete = this._parser[NodeHTTPParser.kOnHeadersComplete] =
  function(majorVersion, minorVersion, headers, method, pathname, statusCode) {
    var info = arguments[0];

    if (typeof info === 'object') {
      method     = info.method;
      pathname   = info.url;
      statusCode = info.statusCode;
      headers    = info.headers;
    }

    self.method     = (typeof method === 'number') ? HttpParser.METHODS[method] : method;
    self.statusCode = statusCode;
    self.url        = pathname;

    if (!headers) return;

    for (var i = 0, n = headers.length, key, value; i < n; i += 2) {
      key   = headers[i].toLowerCase();
      value = headers[i+1];
      if (self.headers.hasOwnProperty(key))
        self.headers[key] += ', ' + value;
      else
        self.headers[key] = value;
    }

    self._complete = true;
  };
};

HttpParser.METHODS = {
  0:  'DELETE',
  1:  'GET',
  2:  'HEAD',
  3:  'POST',
  4:  'PUT',
  5:  'CONNECT',
  6:  'OPTIONS',
  7:  'TRACE',
  8:  'COPY',
  9:  'LOCK',
  10: 'MKCOL',
  11: 'MOVE',
  12: 'PROPFIND',
  13: 'PROPPATCH',
  14: 'SEARCH',
  15: 'UNLOCK',
  16: 'REPORT',
  17: 'MKACTIVITY',
  18: 'CHECKOUT',
  19: 'MERGE',
  24: 'PATCH'
};

HttpParser.prototype.isComplete = function() {
  return this._complete;
};

HttpParser.prototype.parse = function(data) {
  var offset   = (version < 6) ? 1 : 0,
      consumed = this._parser.execute(data, 0, data.length) + offset;

  if (this._complete)
    this.body = (consumed < data.length)
              ? data.slice(consumed)
              : new Buffer(0);
};

module.exports = HttpParser;
