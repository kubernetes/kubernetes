// Copyright 2013 Selenium committers
// Copyright 2013 Software Freedom Conservancy
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//     You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

'use strict';

var assert = require('assert'),
    http = require('http'),
    url = require('url');

var portprober = require('../../net/portprober'),
    promise = require('../..').promise;



/**
 * Encapsulates a simple HTTP server for testing. The {@code onrequest}
 * function should be overridden to define request handling behavior.
 * @param {function(!http.ServerRequest, !http.ServerResponse)} requestHandler
 *     The request handler for the server.
 * @constructor
 */
var Server = function(requestHandler) {
  var server = http.createServer(function(req, res) {
    requestHandler(req, res);
  });

  server.on('connection', function(stream) {
    stream.setTimeout(4000);
  });

  /** @typedef {{port: number, address: string, family: string}} */
  var Host;

  /**
   * Starts the server on the given port. If no port, or 0, is provided,
   * the server will be started on a random port.
   * @param {number=} opt_port The port to start on.
   * @return {!webdriver.promise.Promise.<Host>} A promise that will resolve
   *     with the server host when it has fully started.
   */
  this.start = function(opt_port) {
    var port = opt_port || portprober.findFreePort('localhost');
    return promise.when(port, function(port) {
      return promise.checkedNodeCall(
          server.listen.bind(server, port, 'localhost'));
    }).then(function() {
      return server.address();
    });
  };

  /**
   * Stops the server.
   * @return {!webdriver.promise.Promise} A promise that will resolve when the
   *     server has closed all connections.
   */
  this.stop = function() {
    var d = promise.defer();
    server.close(d.fulfill);
    return d.promise;
  };

  /**
   * @return {Host} This server's host info.
   * @throws {Error} If the server is not running.
   */
  this.address = function() {
    var addr = server.address();
    if (!addr) {
      throw Error('There server is not running!');
    }
    return addr;
  };

  /**
   * return {string} The host:port of this server.
   * @throws {Error} If the server is not running.
   */
  this.host = function() {
    var addr = this.address();
    return addr.address + ':' + addr.port;
  };

  /**
   * Formats a URL for this server.
   * @param {string=} opt_pathname The desired pathname on the server.
   * @return {string} The formatted URL.
   * @throws {Error} If the server is not running.
   */
  this.url = function(opt_pathname) {
    var addr = this.address();
    var pathname = opt_pathname || '';
    return url.format({
      protocol: 'http',
      hostname: addr.address,
      port: addr.port,
      pathname: pathname
    });
  };
};


// PUBLIC API


exports.Server = Server;
