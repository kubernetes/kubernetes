var util   = require('util');
var Server = require('./server');
var Client = require('./client');
var debug  = require('debug')('tinylr');

// Need to keep track of LR servers when notifying
var servers = [];

module.exports = tinylr;

// Expose Server / Client objects
tinylr.Server = Server;
tinylr.Client = Client;

// and the middleware helpers
tinylr.middleware = middleware;
tinylr.changed = changed;

// Main entry point
function tinylr(opts) {
  var srv = new Server(opts);
  servers.push(srv);
  return srv;
}

// A facade to Server#handle
function middleware(opts) {
  var srv = new Server(opts);
  servers.push(srv);
  return function tinylr(req, res, next) {
    srv.handler(req, res, next);
  };
}

// Changed helper, helps with notifying the server of a file change
function changed(done) {
  var files = [].slice.call(arguments);
  if (files[files.length - 1] === 'function') done = files.pop();
  done = typeof done === 'function' ? done : function() {};
  debug('Notifying %d servers - Files: ', servers.length, files);
  servers.forEach(function(srv) {
    var params = { params: { files: files }};
    srv && srv.changed(params);
  });
  done();
}
