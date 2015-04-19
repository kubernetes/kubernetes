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

var fs = require('fs'),
    util = require('util');

var webdriver = require('./index'),
    executors = require('./executors'),
    io = require('./io'),
    portprober = require('./net/portprober'),
    remote = require('./remote');


/**
 * Name of the PhantomJS executable.
 * @type {string}
 * @const
 */
var PHANTOMJS_EXE =
    process.platform === 'win32' ? 'phantomjs.exe' : 'phantomjs';


/**
 * Capability that designates the location of the PhantomJS executable to use.
 * @type {string}
 * @const
 */
var BINARY_PATH_CAPABILITY = 'phantomjs.binary.path';


/**
 * Capability that designates the CLI arguments to pass to PhantomJS.
 * @type {string}
 * @const
 */
var CLI_ARGS_CAPABILITY = 'phantomjs.cli.args';


/**
 * Default log file to use if one is not specified through CLI args.
 * @type {string}
 * @const
 */
var DEFAULT_LOG_FILE = 'phantomjsdriver.log';


/**
 * Finds the PhantomJS executable.
 * @param {string=} opt_exe Path to the executable to use.
 * @return {string} The located executable.
 * @throws {Error} If the executable cannot be found on the PATH, or if the
 *     provided executable path does not exist.
 */
function findExecutable(opt_exe) {
  var exe = opt_exe || io.findInPath(PHANTOMJS_EXE, true);
  if (!exe) {
    throw Error(
        'The PhantomJS executable could not be found on the current PATH. ' +
        'Please download the latest version from ' +
        'http://phantomjs.org/download.html and ensure it can be found on ' +
        'your PATH. For more information, see ' +
        'https://github.com/ariya/phantomjs/wiki');
  }
  if (!fs.existsSync(exe)) {
    throw Error('File does not exist: ' + exe);
  }
  return exe;
}


/**
 * Maps WebDriver logging level name to those recognised by PhantomJS.
 * @type {!Object.<string, string>}
 * @const
 */
var WEBDRIVER_TO_PHANTOMJS_LEVEL = (function() {
  var map = {};
  map[webdriver.logging.Level.ALL.name] = 'DEBUG';
  map[webdriver.logging.Level.DEBUG.name] = 'DEBUG';
  map[webdriver.logging.Level.INFO.name] = 'INFO';
  map[webdriver.logging.Level.WARNING.name] = 'WARN';
  map[webdriver.logging.Level.SEVERE.name] = 'ERROR';
  return map;
})();


/**
 * Creates a new PhantomJS WebDriver client.
 * @param {webdriver.Capabilities=} opt_capabilities The desired capabilities.
 * @param {webdriver.promise.ControlFlow=} opt_flow The control flow to use, or
 *     {@code null} to use the currently active flow.
 * @return {!webdriver.WebDriver} A new WebDriver instance.
 * @deprecated Use {@link Driver}.
 */
function createDriver(opt_capabilities, opt_flow) {
  return new Driver(opt_capabilities, opt_flow);
}


/**
 * Creates a new WebDriver client for PhantomJS.
 *
 * @param {webdriver.Capabilities=} opt_capabilities The desired capabilities.
 * @param {webdriver.promise.ControlFlow=} opt_flow The control flow to use, or
 *     {@code null} to use the currently active flow.
 * @constructor
 * @extends {webdriver.WebDriver}
 */
var Driver = function(opt_capabilities, opt_flow) {
  var capabilities = opt_capabilities || webdriver.Capabilities.phantomjs();
  var exe = findExecutable(capabilities.get(BINARY_PATH_CAPABILITY));
  var args = ['--webdriver-logfile=' + DEFAULT_LOG_FILE];

  var logPrefs = capabilities.get(webdriver.Capability.LOGGING_PREFS);
  if (logPrefs instanceof webdriver.logging.Preferences) {
    logPrefs = logPrefs.toJSON();
  }

  if (logPrefs && logPrefs[webdriver.logging.Type.DRIVER]) {
    var level = WEBDRIVER_TO_PHANTOMJS_LEVEL[
        logPrefs[webdriver.logging.Type.DRIVER]];
    if (level) {
      args.push('--webdriver-loglevel=' + level);
    }
  }

  var proxy = capabilities.get(webdriver.Capability.PROXY);
  if (proxy) {
    switch (proxy.proxyType) {
      case 'manual':
        if (proxy.httpProxy) {
          args.push(
              '--proxy-type=http',
              '--proxy=http://' + proxy.httpProxy);
        }
        break;
      case 'pac':
        throw Error('PhantomJS does not support Proxy PAC files');
      case 'system':
        args.push('--proxy-type=system');
        break;
      case 'direct':
        args.push('--proxy-type=none');
        break;
    }
  }
  args = args.concat(capabilities.get(CLI_ARGS_CAPABILITY) || []);

  var port = portprober.findFreePort();
  var service = new remote.DriverService(exe, {
    port: port,
    args: webdriver.promise.when(port, function(port) {
      args.push('--webdriver=' + port);
      return args;
    })
  });

  var executor = executors.createExecutor(service.start());
  var driver = webdriver.WebDriver.createSession(
      executor, capabilities, opt_flow);

  webdriver.WebDriver.call(
      this, driver.getSession(), executor, driver.controlFlow());

  var boundQuit = this.quit.bind(this);

  /** @override */
  this.quit = function() {
    return boundQuit().thenFinally(service.kill.bind(service));
  };
  return driver;
};
util.inherits(Driver, webdriver.WebDriver);


// PUBLIC API

exports.Driver = Driver;
exports.createDriver = createDriver;
