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
 * Name of the ChromeDriver executable.
 * @type {string}
 * @const
 */
var CHROMEDRIVER_EXE =
    process.platform === 'win32' ? 'chromedriver.exe' : 'chromedriver';


/**
 * Creates {@link remote.DriverService} instances that manage a ChromeDriver
 * server.
 * @param {string=} opt_exe Path to the server executable to use. If omitted,
 *     the builder will attempt to locate the chromedriver on the current
 *     PATH.
 * @throws {Error} If provided executable does not exist, or the chromedriver
 *     cannot be found on the PATH.
 * @constructor
 */
var ServiceBuilder = function(opt_exe) {
  /** @private {string} */
  this.exe_ = opt_exe || io.findInPath(CHROMEDRIVER_EXE, true);
  if (!this.exe_) {
    throw Error(
        'The ChromeDriver could not be found on the current PATH. Please ' +
        'download the latest version of the ChromeDriver from ' +
        'http://chromedriver.storage.googleapis.com/index.html and ensure ' +
        'it can be found on your PATH.');
  }

  if (!fs.existsSync(this.exe_)) {
    throw Error('File does not exist: ' + this.exe_);
  }

  /** @private {!Array.<string>} */
  this.args_ = [];
  this.stdio_ = 'ignore';
};


/** @private {number} */
ServiceBuilder.prototype.port_ = 0;


/** @private {(string|!Array.<string|number|!Stream|null|undefined>)} */
ServiceBuilder.prototype.stdio_ = 'ignore';


/** @private {Object.<string, string>} */
ServiceBuilder.prototype.env_ = null;


/**
 * Sets the port to start the ChromeDriver on.
 * @param {number} port The port to use, or 0 for any free port.
 * @return {!ServiceBuilder} A self reference.
 * @throws {Error} If the port is invalid.
 */
ServiceBuilder.prototype.usingPort = function(port) {
  if (port < 0) {
    throw Error('port must be >= 0: ' + port);
  }
  this.port_ = port;
  return this;
};


/**
 * Sets the path of the log file the driver should log to. If a log file is
 * not specified, the driver will log to stderr.
 * @param {string} path Path of the log file to use.
 * @return {!ServiceBuilder} A self reference.
 */
ServiceBuilder.prototype.loggingTo = function(path) {
  this.args_.push('--log-path=' + path);
  return this;
};


/**
 * Enables verbose logging.
 * @return {!ServiceBuilder} A self reference.
 */
ServiceBuilder.prototype.enableVerboseLogging = function() {
  this.args_.push('--verbose');
  return this;
};


/**
 * Sets the number of threads the driver should use to manage HTTP requests.
 * By default, the driver will use 4 threads.
 * @param {number} n The number of threads to use.
 * @return {!ServiceBuilder} A self reference.
 */
ServiceBuilder.prototype.setNumHttpThreads = function(n) {
  this.args_.push('--http-threads=' + n);
  return this;
};


/**
 * Sets the base path for WebDriver REST commands (e.g. "/wd/hub").
 * By default, the driver will accept commands relative to "/".
 * @param {string} path The base path to use.
 * @return {!ServiceBuilder} A self reference.
 */
ServiceBuilder.prototype.setUrlBasePath = function(path) {
  this.args_.push('--url-base=' + path);
  return this;
};


/**
 * Defines the stdio configuration for the driver service. See
 * {@code child_process.spawn} for more information.
 * @param {(string|!Array.<string|number|!Stream|null|undefined>)} config The
 *     configuration to use.
 * @return {!ServiceBuilder} A self reference.
 */
ServiceBuilder.prototype.setStdio = function(config) {
  this.stdio_ = config;
  return this;
};


/**
 * Defines the environment to start the server under. This settings will be
 * inherited by every browser session started by the server.
 * @param {!Object.<string, string>} env The environment to use.
 * @return {!ServiceBuilder} A self reference.
 */
ServiceBuilder.prototype.withEnvironment = function(env) {
  this.env_ = env;
  return this;
};


/**
 * Creates a new DriverService using this instance's current configuration.
 * @return {remote.DriverService} A new driver service using this instance's
 *     current configuration.
 * @throws {Error} If the driver exectuable was not specified and a default
 *     could not be found on the current PATH.
 */
ServiceBuilder.prototype.build = function() {
  var port = this.port_ || portprober.findFreePort();
  var args = this.args_.concat();  // Defensive copy.

  return new remote.DriverService(this.exe_, {
    loopback: true,
    port: port,
    args: webdriver.promise.when(port, function(port) {
      return args.concat('--port=' + port);
    }),
    env: this.env_,
    stdio: this.stdio_
  });
};


/** @type {remote.DriverService} */
var defaultService = null;


/**
 * Sets the default service to use for new ChromeDriver instances.
 * @param {!remote.DriverService} service The service to use.
 * @throws {Error} If the default service is currently running.
 */
function setDefaultService(service) {
  if (defaultService && defaultService.isRunning()) {
    throw Error(
        'The previously configured ChromeDriver service is still running. ' +
        'You must shut it down before you may adjust its configuration.');
  }
  defaultService = service;
}


/**
 * Returns the default ChromeDriver service. If such a service has not been
 * configured, one will be constructed using the default configuration for
 * a ChromeDriver executable found on the system PATH.
 * @return {!remote.DriverService} The default ChromeDriver service.
 */
function getDefaultService() {
  if (!defaultService) {
    defaultService = new ServiceBuilder().build();
  }
  return defaultService;
}


/**
 * @type {string}
 * @const
 */
var OPTIONS_CAPABILITY_KEY = 'chromeOptions';


/**
 * Class for managing ChromeDriver specific options.
 * @constructor
 */
var Options = function() {
  /** @private {!Array.<string>} */
  this.args_ = [];

  /** @private {!Array.<(string|!Buffer)>} */
  this.extensions_ = [];
};


/**
 * Extracts the ChromeDriver specific options from the given capabilities
 * object.
 * @param {!webdriver.Capabilities} capabilities The capabilities object.
 * @return {!Options} The ChromeDriver options.
 */
Options.fromCapabilities = function(capabilities) {
  var options = new Options();

  var o = capabilities.get(OPTIONS_CAPABILITY_KEY);
  if (o instanceof Options) {
    options = o;
  } else if (o) {
    options.
        addArguments(o.args || []).
        addExtensions(o.extensions || []).
        detachDriver(!!o.detach).
        setChromeBinaryPath(o.binary).
        setChromeLogFile(o.logFile).
        setLocalState(o.localState).
        setUserPreferences(o.prefs);
  }

  if (capabilities.has(webdriver.Capability.PROXY)) {
    options.setProxy(capabilities.get(webdriver.Capability.PROXY));
  }

  if (capabilities.has(webdriver.Capability.LOGGING_PREFS)) {
    options.setLoggingPrefs(
        capabilities.get(webdriver.Capability.LOGGING_PREFS));
  }

  return options;
};


/**
 * Add additional command line arguments to use when launching the Chrome
 * browser.  Each argument may be specified with or without the "--" prefix
 * (e.g. "--foo" and "foo"). Arguments with an associated value should be
 * delimited by an "=": "foo=bar".
 * @param {...(string|!Array.<string>)} var_args The arguments to add.
 * @return {!Options} A self reference.
 */
Options.prototype.addArguments = function(var_args) {
  this.args_ = this.args_.concat.apply(this.args_, arguments);
  return this;
};


/**
 * Add additional extensions to install when launching Chrome. Each extension
 * should be specified as the path to the packed CRX file, or a Buffer for an
 * extension.
 * @param {...(string|!Buffer|!Array.<(string|!Buffer)>)} var_args The
 *     extensions to add.
 * @return {!Options} A self reference.
 */
Options.prototype.addExtensions = function(var_args) {
  this.extensions_ = this.extensions_.concat.apply(
      this.extensions_, arguments);
  return this;
};


/**
 * Sets the path to the Chrome binary to use. On Mac OS X, this path should
 * reference the actual Chrome executable, not just the application binary
 * (e.g. "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome").
 *
 * The binary path be absolute or relative to the chromedriver server
 * executable, but it must exist on the machine that will launch Chrome.
 *
 * @param {string} path The path to the Chrome binary to use.
 * @return {!Options} A self reference.
 */
Options.prototype.setChromeBinaryPath = function(path) {
  this.binary_ = path;
  return this;
};


/**
 * Sets whether to leave the started Chrome browser running if the controlling
 * ChromeDriver service is killed before {@link webdriver.WebDriver#quit()} is
 * called.
 * @param {boolean} detach Whether to leave the browser running if the
 *     chromedriver service is killed before the session.
 * @return {!Options} A self reference.
 */
Options.prototype.detachDriver = function(detach) {
  this.detach_ = detach;
  return this;
};


/**
 * Sets the user preferences for Chrome's user profile. See the "Preferences"
 * file in Chrome's user data directory for examples.
 * @param {!Object} prefs Dictionary of user preferences to use.
 * @return {!Options} A self reference.
 */
Options.prototype.setUserPreferences = function(prefs) {
  this.prefs_ = prefs;
  return this;
};


/**
 * Sets the logging preferences for the new session.
 * @param {!webdriver.logging.Preferences} prefs The logging preferences.
 * @return {!Options} A self reference.
 */
Options.prototype.setLoggingPrefs = function(prefs) {
  this.logPrefs_ = prefs;
  return this;
};


/**
 * Sets preferences for the "Local State" file in Chrome's user data
 * directory.
 * @param {!Object} state Dictionary of local state preferences.
 * @return {!Options} A self reference.
 */
Options.prototype.setLocalState = function(state) {
  this.localState_ = state;
  return this;
};


/**
 * Sets the path to Chrome's log file. This path should exist on the machine
 * that will launch Chrome.
 * @param {string} path Path to the log file to use.
 * @return {!Options} A self reference.
 */
Options.prototype.setChromeLogFile = function(path) {
  this.logFile_ = path;
  return this;
};


/**
 * Sets the proxy settings for the new session.
 * @param {webdriver.ProxyConfig} proxy The proxy configuration to use.
 * @return {!Options} A self reference.
 */
Options.prototype.setProxy = function(proxy) {
  this.proxy_ = proxy;
  return this;
};


/**
 * Converts this options instance to a {@link webdriver.Capabilities} object.
 * @param {webdriver.Capabilities=} opt_capabilities The capabilities to merge
 *     these options into, if any.
 * @return {!webdriver.Capabilities} The capabilities.
 */
Options.prototype.toCapabilities = function(opt_capabilities) {
  var capabilities = opt_capabilities || webdriver.Capabilities.chrome();
  capabilities.
      set(webdriver.Capability.PROXY, this.proxy_).
      set(webdriver.Capability.LOGGING_PREFS, this.logPrefs_).
      set(OPTIONS_CAPABILITY_KEY, this);
  return capabilities;
};


/**
 * Converts this instance to its JSON wire protocol representation. Note this
 * function is an implementation not intended for general use.
 * @return {{args: !Array.<string>,
 *           binary: (string|undefined),
 *           detach: boolean,
 *           extensions: !Array.<string>,
 *           localState: (Object|undefined),
 *           logFile: (string|undefined),
 *           prefs: (Object|undefined)}} The JSON wire protocol representation
 *     of this instance.
 */
Options.prototype.toJSON = function() {
  return {
    args: this.args_,
    binary: this.binary_,
    detach: !!this.detach_,
    extensions: this.extensions_.map(function(extension) {
      if (Buffer.isBuffer(extension)) {
        return extension.toString('base64');
      }
      return fs.readFileSync(extension, 'base64');
    }),
    localState: this.localState_,
    logFile: this.logFile_,
    prefs: this.prefs_
  };
};


/**
 * Creates a new ChromeDriver session.
 * @param {(webdriver.Capabilities|Options)=} opt_options The session options.
 * @param {remote.DriverService=} opt_service The session to use; will use
 *     the {@link getDefaultService default service} by default.
 * @param {webdriver.promise.ControlFlow=} opt_flow The control flow to use, or
 *     {@code null} to use the currently active flow.
 * @return {!webdriver.WebDriver} A new WebDriver instance.
 * @deprecated Use {@link Driver new Driver()}.
 */
function createDriver(opt_options, opt_service, opt_flow) {
  return new Driver(opt_options, opt_service, opt_flow);
}


/**
 * Creates a new WebDriver client for Chrome.
 *
 * @param {(webdriver.Capabilities|Options)=} opt_config The configuration
 *     options.
 * @param {remote.DriverService=} opt_service The session to use; will use
 *     the {@link getDefaultService default service} by default.
 * @param {webdriver.promise.ControlFlow=} opt_flow The control flow to use, or
 *     {@code null} to use the currently active flow.
 * @constructor
 * @extends {webdriver.WebDriver}
 */
var Driver = function(opt_config, opt_service, opt_flow) {
  var service = opt_service || getDefaultService();
  var executor = executors.createExecutor(service.start());

  var capabilities =
      opt_config instanceof Options ? opt_config.toCapabilities() :
      (opt_config || webdriver.Capabilities.chrome());

  var driver = webdriver.WebDriver.createSession(
      executor, capabilities, opt_flow);

  webdriver.WebDriver.call(
      this, driver.getSession(), executor, driver.controlFlow());
};
util.inherits(Driver, webdriver.WebDriver);


// PUBLIC API


exports.Driver = Driver;
exports.Options = Options;
exports.ServiceBuilder = ServiceBuilder;
exports.createDriver = createDriver;
exports.getDefaultService = getDefaultService;
exports.setDefaultService = setDefaultService;
