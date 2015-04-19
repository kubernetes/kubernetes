'use strict';

var gutil = require('gulp-util'),
    path = require('path'),
    tinylr = require('tiny-lr'),
    merge = require('lodash.assign'),
    Transform = require('stream').Transform,
    magenta = gutil.colors.magenta;

module.exports = exports = function (server, opts) {
  var reload = new Transform({ objectMode:true });

  if (server !== null &&
      typeof server === 'object' &&
      !(server instanceof tinylr.Server) &&
      !opts) {
    merge(exports.options, server);
    server = null;
  } else {
    merge(exports.options, opts);
  }

  reload._transform = function(file, encoding, next) {
    exports.changed(file.path, server);
    this.push(file);
    next();
  };

  reload.changed = exports.changed;

  return reload;
};

exports.options = { auto: true };
exports.servers = {};

/**
 * lr.listen()
 * lr.listen(server)
 * lr.listen(port)
 */

exports.listen = function(server, opts) {
  if (server !== null &&
      typeof server === 'object' &&
      !(server instanceof tinylr.Server) &&
      !opts) {
    merge(exports.options, server);
    server = null;
  } else {
    merge(exports.options, opts);
  }

  server = server || 35729;

  if (typeof server === 'number') {
    var port = server;

    if (exports.servers[port]) {
      return exports.servers[port];
    }

    if (!exports.options.auto) {
      return;
    }

    exports.servers[port] = server = tinylr(exports.options);
    server.listen(port, function (err) {
      if (err) {
        throw new gutil.PluginError('gulp-livereload', err.message);
      }
      if (!exports.options.silent) {
        gutil.log('Live reload server listening on: ' + magenta(port));
      }
    });
  }

  return server;
};

/**
 * lr.changed(filepath)
 * lr.changed(filepath, server)
 * lr.changed(filepath, port)
 */

exports.changed = function(filePath, server) {
  server = exports.listen(server);
  filePath = (filePath) ? filePath.hasOwnProperty('path') ? filePath.path : filePath : '*';

  if (!server) return;

  if (!exports.options.silent) {
    gutil.log(magenta(path.basename(filePath)) + ' was reloaded.');
  }

  server.changed({ body: { files: [filePath] } });
};

exports.middleware = tinylr.middleware;
