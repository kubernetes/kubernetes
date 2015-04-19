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

var path = require('path'),
    url = require('url'),
    util = require('util');

var promise = require('../').promise,
    httpUtil = require('../http/util'),
    exec = require('../io/exec'),
    net = require('../net'),
    portprober = require('../net/portprober');



/**
 * Configuration options for a DriverService instance.
 * <ul>
 * <li>
 * <li>{@code loopback} - Whether the service should only be accessed on this
 *     host's loopback address.
 * <li>{@code port} - The port to start the server on (must be > 0). If the
 *     port is provided as a promise, the service will wait for the promise to
 *     resolve before starting.
 * <li>{@code args} - The arguments to pass to the service. If a promise is
 *     provided, the service will wait for it to resolve before starting.
 * <li>{@code path} - The base path on the server for the WebDriver wire
 *     protocol (e.g. '/wd/hub'). Defaults to '/'.
 * <li>{@code env} - The environment variables that should be visible to the
 *     server process. Defaults to inheriting the current process's
 *     environment.
 * <li>{@code stdio} - IO configuration for the spawned server process. For
 *     more information, refer to the documentation of
 *     {@code child_process.spawn}.
 * </ul>
 *
 * @typedef {{
 *   port: (number|!webdriver.promise.Promise.<number>),
 *   args: !(Array.<string>|webdriver.promise.Promise.<!Array.<string>>),
 *   path: (string|undefined),
 *   env: (!Object.<string, string>|undefined),
 *   stdio: (string|!Array.<string|number|!Stream|null|undefined>|undefined)
 * }}
 */
var ServiceOptions;


/**
 * Manages the life and death of a native executable WebDriver server.
 *
 * <p>It is expected that the driver server implements the
 * <a href="http://code.google.com/p/selenium/wiki/JsonWireProtocol">WebDriver
 * Wire Protocol</a>. Furthermore, the managed server should support multiple
 * concurrent sessions, so that this class may be reused for multiple clients.
 *
 * @param {string} executable Path to the executable to run.
 * @param {!ServiceOptions} options Configuration options for the service.
 * @constructor
 */
function DriverService(executable, options) {

  /** @private {string} */
  this.executable_ = executable;

  /** @private {boolean} */
  this.loopbackOnly_ = !!options.loopback;

  /** @private {(number|!webdriver.promise.Promise.<number>)} */
  this.port_ = options.port;

  /**
   * @private {!(Array.<string>|webdriver.promise.Promise.<!Array.<string>>)}
   */
  this.args_ = options.args;

  /** @private {string} */
  this.path_ = options.path || '/';

  /** @private {!Object.<string, string>} */
  this.env_ = options.env || process.env;

  /** @private {(string|!Array.<string|number|!Stream|null|undefined>)} */
  this.stdio_ = options.stdio || 'ignore';

  /**
   * A promise for the managed subprocess, or null if the server has not been
   * started yet. This promise will never be rejected.
   * @private {promise.Promise.<!exec.Command>}
   */
  this.command_ = null;

  /**
   * Promise that resolves to the server's address or null if the server has
   * not been started. This promise will be rejected if the server terminates
   * before it starts accepting WebDriver requests.
   * @private {promise.Promise.<string>}
   */
  this.address_ = null;
}


/**
 * The default amount of time, in milliseconds, to wait for the server to
 * start.
 * @type {number}
 */
DriverService.DEFAULT_START_TIMEOUT_MS = 30 * 1000;


/**
 * @return {!webdriver.promise.Promise.<string>} A promise that resolves to
 *    the server's address.
 * @throws {Error} If the server has not been started.
 */
DriverService.prototype.address = function() {
  if (this.address_) {
    return this.address_;
  }
  throw Error('Server has not been started.');
};


/**
 * Returns whether the underlying process is still running. This does not take
 * into account whether the process is in the process of shutting down.
 * @return {boolean} Whether the underlying service process is running.
 */
DriverService.prototype.isRunning = function() {
  return !!this.address_;
};


/**
 * Starts the server if it is not already running.
 * @param {number=} opt_timeoutMs How long to wait, in milliseconds, for the
 *     server to start accepting requests. Defaults to 30 seconds.
 * @return {!promise.Promise.<string>} A promise that will resolve
 *     to the server's base URL when it has started accepting requests. If the
 *     timeout expires before the server has started, the promise will be
 *     rejected.
 */
DriverService.prototype.start = function(opt_timeoutMs) {
  if (this.address_) {
    return this.address_;
  }

  var timeout = opt_timeoutMs || DriverService.DEFAULT_START_TIMEOUT_MS;

  var self = this;
  this.command_ = promise.defer();
  this.address_ = promise.defer();
  this.address_.fulfill(promise.when(this.port_, function(port) {
    if (port <= 0) {
      throw Error('Port must be > 0: ' + port);
    }
    return promise.when(self.args_, function(args) {
      var command = exec(self.executable_, {
        args: args,
        env: self.env_,
        stdio: self.stdio_
      });

      self.command_.fulfill(command);

      command.result().then(function(result) {
        self.address_.reject(result.code == null ?
            Error('Server was killed with ' + result.signal) :
            Error('Server exited with ' + result.code));
        self.address_ = null;
        self.command_ = null;
      });

      var serverUrl = url.format({
        protocol: 'http',
        hostname: !self.loopbackOnly_ && net.getAddress() ||
            net.getLoopbackAddress(),
        port: port,
        pathname: self.path_
      });

      return httpUtil.waitForServer(serverUrl, timeout).then(function() {
        return serverUrl;
      });
    });
  }));

  return this.address_;
};


/**
 * Stops the service if it is not currently running. This function will kill
 * the server immediately. To synchronize with the active control flow, use
 * {@link #stop()}.
 * @return {!webdriver.promise.Promise} A promise that will be resolved when
 *     the server has been stopped.
 */
DriverService.prototype.kill = function() {
  if (!this.address_ || !this.command_) {
    return promise.fulfilled();  // Not currently running.
  }
  return this.command_.then(function(command) {
    command.kill('SIGTERM');
  });
};


/**
 * Schedules a task in the current control flow to stop the server if it is
 * currently running.
 * @return {!webdriver.promise.Promise} A promise that will be resolved when
 *     the server has been stopped.
 */
DriverService.prototype.stop = function() {
  return promise.controlFlow().execute(this.kill.bind(this));
};



/**
 * Manages the life and death of the Selenium standalone server. The server
 * may be obtained from http://selenium-release.storage.googleapis.com/index.html.
 * @param {string} jar Path to the Selenium server jar.
 * @param {!SeleniumServer.Options} options Configuration options for the
 *     server.
 * @throws {Error} If an invalid port is specified.
 * @constructor
 * @extends {DriverService}
 */
function SeleniumServer(jar, options) {
  if (options.port < 0)
    throw Error('Port must be >= 0: ' + options.port);

  var port = options.port || portprober.findFreePort();
  var args = promise.when(options.jvmArgs || [], function(jvmArgs) {
    return promise.when(options.args || [], function(args) {
      return promise.when(port, function(port) {
        return jvmArgs.concat(['-jar', jar, '-port', port]).concat(args);
      });
    });
  });

  DriverService.call(this, 'java', {
    port: port,
    args: args,
    path: '/wd/hub',
    env: options.env,
    stdio: options.stdio
  });
}
util.inherits(SeleniumServer, DriverService);


/**
 * Options for the Selenium server:
 * <ul>
 * <li>{@code port} - The port to start the server on (must be > 0). If the
 *     port is provided as a promise, the service will wait for the promise to
 *     resolve before starting.
 * <li>{@code args} - The arguments to pass to the service. If a promise is
 *     provided, the service will wait for it to resolve before starting.
 * <li>{@code jvmArgs} - The arguments to pass to the JVM. If a promise is
 *     provided, the service will wait for it to resolve before starting.
 * <li>{@code env} - The environment variables that should be visible to the
 *     server process. Defaults to inheriting the current process's
 *     environment.
 * <li>{@code stdio} - IO configuration for the spawned server process. For
 *     more information, refer to the documentation of
 *     {@code child_process.spawn}.
 * </ul>
 *
 * @typedef {{
 *   port: (number|!webdriver.promise.Promise.<number>),
 *   args: !(Array.<string>|webdriver.promise.Promise.<!Array.<string>>),
 *   jvmArgs: (!Array.<string>|
 *             !webdriver.promise.Promise.<!Array.<string>>|
 *             undefined),
 *   env: (!Object.<string, string>|undefined),
 *   stdio: (string|!Array.<string|number|!Stream|null|undefined>|undefined)
 * }}
 */
SeleniumServer.Options;


// PUBLIC API

exports.DriverService = DriverService;
exports.SeleniumServer = SeleniumServer;
