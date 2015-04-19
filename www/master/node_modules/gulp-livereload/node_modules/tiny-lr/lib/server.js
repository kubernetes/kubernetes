var fs          = require('fs');
var qs          = require('qs');
var path        = require('path');
var util        = require('util');
var http        = require('http');
var https       = require('https');
var events      = require('events');
var parse       = require('url').parse;
var debug       = require('debug')('tinylr:server');
var Client      = require('./client');
var constants   = require('constants');

// Middleware fallbacks
var bodyParser  = require('body-parser').json()
var queryParser = require('./middleware/query')();

var config = require('../package.json');

function Server(options) {
  options = this.options = options || {};
  events.EventEmitter.call(this);

  options.livereload = options.livereload || path.join(__dirname, '../node_modules/livereload-js/dist/livereload.js');
  options.port = parseInt(options.port || 35729, 10);

  this.on('GET /', this.index.bind(this));
  this.on('GET /changed', this.changed.bind(this));
  this.on('POST /changed', this.changed.bind(this));
  this.on('GET /livereload.js', this.livereload.bind(this));
  this.on('GET /kill', this.close.bind(this));

  this.clients = {};
  this.configure(options.app);
}

module.exports = Server;

util.inherits(Server, events.EventEmitter);

Server.prototype.configure = function configure(app) {
  var self = this;
  debug('Configuring %s', app ? 'connect / express application' : 'HTTP server');

  if (!app) {
    if ((this.options.key && this.options.cert) || this.options.pfx) {
      this.server = https.createServer(this.options, this.handler.bind(this));
    } else {
      this.server = http.createServer(this.handler.bind(this));
    }
    this.server.on('upgrade', this.websocketify.bind(this));
    this.server.on('error', function() {
      self.error.apply(self, arguments);
    });
    return this;
  }

  this.app = app;

  this.app.listen = function(port, done) {
    done = done || function() {};
    if (port !== self.options.port) {
      debug('Warn: LiveReload port is not standard (%d). You are listening on %d', self.options.port, port);
      debug('You\'ll need to rely on the LiveReload snippet');
      debug('> http://feedback.livereload.com/knowledgebase/articles/86180-how-do-i-add-the-script-tag-manually-');
    }

    var srv = self.server = http.createServer(app);
    srv.on('upgrade', self.websocketify.bind(self));
    srv.on('error', function() {
      self.error.apply(self, arguments);
    });
    srv.on('close', self.close.bind(self));
    return srv.listen(port, done);
  };

  return this;
};

Server.prototype.handler = function handler(req, res, next) {
  var self = this;
  var middleware = typeof next === 'function';
  debug('LiveReload handler %s (middleware: %s)', req.url, middleware ? 'on' : 'off');

  this.parse(req, res, function(err) {
    debug('query parsed', req.body, err);
    if (err) return next(err);
    self.handle(req, res, next);
  });

  // req
  //   .on('end', this.handle.bind(this, req, res))
  //   .on('data', function(chunk) {
  //     req.data = req.data || '';
  //     req.data += chunk;
  //   });

  return this;
};

// Ensure body / query are defined, useful as a fallback when the
// Server is used without express / connect, and shouldn't hurt
// otherwise
Server.prototype.parse = function(req, res, next) {
  debug('Parse', req.body, req.query);
  bodyParser(req, res, function(err) {
    debug('Body parsed', req.body);
    if (err) return next(err);

    queryParser(req, res, next);
  });
};

Server.prototype.handle = function handle(req, res, next) {
  var url = parse(req.url);
  debug('Request:', req.method, url.href);
  var middleware = typeof next === 'function';

  res.setHeader('Content-Type', 'application/json');

  // do the routing
  var route = req.method + ' '  + url.pathname;
  var respond = this.emit(route, req, res);
  if (respond) return;
  if (middleware) return next();

  res.writeHead(404);
  res.write(JSON.stringify({
    error: 'not_found',
    reason: 'no such route'
  }));
  res.end();
};

Server.prototype.websocketify = function websocketify(req, socket, head) {
  var self = this;
  var client = new Client(req, socket, head, this.options);
  this.clients[client.id] = client;

  debug('New LiveReload connection (id: %s)', client.id);
  client.on('end', function() {
    debug('Destroy client %s (url: %s)', client.id, client.url);
    delete self.clients[client.id];
  });
};

Server.prototype.listen = function listen(port, host, fn) {
  this.port = port;

  if (typeof host === 'function') {
    fn = host;
    host = undefined;
  }

  this.server.listen(port, host, fn);
};

Server.prototype.close = function close(req, res) {
  Object.keys(this.clients).forEach(function(id) {
    this.clients[id].close();
  }, this);


  if (this.server._handle) this.server.close(this.emit.bind(this, 'close'));

  if (res) res.end();
};

Server.prototype.error = function error(e) {
  console.error();
  console.error('... Uhoh. Got error %s ...', e.message);
  console.error(e.stack);

  if (e.code !== constants.EADDRINUSE) return;
  console.error();
  console.error('You already have a server listening on %s', this.port);
  console.error('You should stop it and try again.');
  console.error();
};

// Routes

Server.prototype.livereload = function livereload(req, res) {
  fs.createReadStream(this.options.livereload).pipe(res);
};

Server.prototype.changed = function changed(req, res) {
  var files = this.param('files', req);

  debug('Changed event (Files: %s)', files.join(' '));
  var clients = this.notifyClients(files);

  if (!res) return;

  res.write(JSON.stringify({
    clients: clients,
    files: files
  }));

  res.end();
};

Server.prototype.notifyClients = function notifyClients(files) {
  var clients = Object.keys(this.clients).map(function(id) {
    var client = this.clients[id];
    debug('Reloading client %s (url: %s)', client.id, client.url);
    client.reload(files);
    return {
      id: client.id,
      url: client.url
    };
  }, this);

  return clients;
};

// Lookup param from body / params / query.
Server.prototype.param = function _param(name, req) {
  var param;
  if (req.body && req.body[name]) param = req.body.files;
  else if (req.params && req.params[name]) param = req.params.files;
  else if (req.query && req.query[name]) param= req.query.files;

  // normalize files array
  param = Array.isArray(param) ? param :
    typeof param === 'string' ? param.split(/[\s,]/) :
    [];

  debug('param %s', name, req.body, req.params, req.query, param);
  return param;
};

Server.prototype.index = function index(req, res) {
  res.write(JSON.stringify({
    tinylr: 'Welcome',
    version: config.version
  }));

  res.end();
};
