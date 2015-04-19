/*
 * index.js: Top level include for node-http-proxy helpers
 *
 * (C) 2010 Nodejitsu Inc.
 * MIT LICENCE
 *
 */

var fs = require('fs'),
    path = require('path');

var fixturesDir = path.join(__dirname, '..', 'fixtures');

//
// @https {Object}
// Returns the necessary `https` credentials.
//
Object.defineProperty(exports, 'https', {
  get: function () {
    delete this.https;
    return this.https = {
      key:  fs.readFileSync(path.join(fixturesDir, 'agent2-key.pem'), 'utf8'),
      cert: fs.readFileSync(path.join(fixturesDir, 'agent2-cert.pem'), 'utf8')
    };
  }
});

//
// @protocols {Object}
// Returns an object representing the desired protocols
// for the `proxy` and `target` server.
//
Object.defineProperty(exports, 'protocols', {
  get: function () {
    delete this.protocols;
    return this.protocols = {
      target: exports.argv.target || 'http',
      proxy: exports.argv.proxy || 'http'
    };
  }
});

//
// @nextPort {number}
// Returns an auto-incrementing port for tests.
//
Object.defineProperty(exports, 'nextPort', {
  get: function () {
    var current = this.port || 9050;
    this.port = current + 1;
    return current;
  }
});

//
// @nextPortPair {Object}
// Returns an auto-incrementing pair of ports for tests.
//
Object.defineProperty(exports, 'nextPortPair', {
  get: function () {
    return {
      target: this.nextPort,
      proxy: this.nextPort
    };
  }
});

//
// ### function describe(prefix)
// #### @prefix {string} Prefix to use before the description
//
// Returns a string representing the protocols that this suite
// is testing based on CLI arguments.
//
exports.describe = function (prefix, base) {
  prefix = prefix || '';
  base   = base   || 'http';

  function protocol(endpoint) {
    return exports.protocols[endpoint] === 'https'
      ? base + 's'
      : base;
  }

  return [
    'node-http-proxy',
    prefix,
    [
      protocol('proxy'),
      '-to-',
      protocol('target')
    ].join('')
  ].filter(Boolean).join('/');
};

//
// Expose the CLI arguments
//
exports.argv = require('optimist').argv;

//
// Export additional helpers for `http` and `websockets`.
//
exports.http = require('./http');
exports.ws   = require('./ws');