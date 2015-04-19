// Copyright 2011 Software Freedom Conservancy. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

var base = require('./_base'),
    executors = require('./executors');

// Use base.require to avoid circular references between index and this module.
var Browser = base.require('webdriver.Browser'),
    Capabilities = base.require('webdriver.Capabilities'),
    Capability = base.require('webdriver.Capability'),
    WebDriver = base.require('webdriver.WebDriver'),
    promise = base.require('webdriver.promise');



/**
 * Creates new {@link webdriver.WebDriver WebDriver} instances. The environment
 * variables listed below may be used to override a builder's configuration,
 * allowing quick runtime changes.
 * <ul>
 * <li>{@code SELENIUM_REMOTE_URL}: defines the remote URL for all builder
 *   instances. This environment variable should be set to a fully qualified
 *   URL for a WebDriver server (e.g. http://localhost:4444/wd/hub).
 *
 * <li>{@code SELENIUM_BROWSER}: defines the target browser in the form
 *   {@code browser[:version][:platform]}.
 * </ul>
 *
 * <p>Suppose you had mytest.js that created WebDriver with
 *     {@code var driver = new webdriver.Builder().build();}.
 *
 * This test could be made to use Firefox on the local machine by running with
 * {@code SELENIUM_BROWSER=firefox node mytest.js}.
 *
 * <p>Alternatively, you could request Chrome 36 on Linux from a remote
 * server with {@code
 *     SELENIUM_BROWSER=chrome:36:LINUX
 *     SELENIUM_REMOTE_URL=http://www.example.com:4444/wd/hub
 *     node mytest.js}.
 *
 * @constructor
 */
var Builder = function() {

  /** @private {webdriver.promise.ControlFlow} */
  this.flow_ = null;

  /** @private {string} */
  this.url_ = '';

  /** @private {!webdriver.Capabilities} */
  this.capabilities_ = new Capabilities();

  /** @private {chrome.Options} */
  this.chromeOptions_ = null;

  /** @private {firefox.Options} */
  this.firefoxOptions_ = null;
};


/**
 * Sets the URL of a remote WebDriver server to use. Once a remote URL has been
 * specified, the builder direct all new clients to that server. If this method
 * is never called, the Builder will attempt to create all clients locally.
 *
 * <p>As an alternative to this method, you may also set the
 * {@code SELENIUM_REMOTE_URL} environment variable.
 *
 * @param {string} url The URL of a remote server to use.
 * @return {!Builder} A self reference.
 */
Builder.prototype.usingServer = function(url) {
  this.url_ = url;
  return this;
};


/**
 * @return {string} The URL of the WebDriver server this instance is configured
 *     to use.
 */
Builder.prototype.getServerUrl = function() {
  return this.url_;
};


/**
 * Sets the desired capabilities when requesting a new session. This will
 * overwrite any previously set capabilities.
 * @param {!(Object|webdriver.Capabilities)} capabilities The desired
 *     capabilities for a new session.
 * @return {!Builder} A self reference.
 */
Builder.prototype.withCapabilities = function(capabilities) {
  this.capabilities_ = new Capabilities(capabilities);
  return this;
};


/**
 * Returns the base set of capabilities this instance is currently configured
 * to use.
 * @return {!webdriver.Capabilities} The current capabilities for this builder.
 */
Builder.prototype.getCapabilities = function() {
  return this.capabilities_;
};


/**
 * Configures the target browser for clients created by this instance.
 * Any calls to {@link #withCapabilities} after this function will
 * overwrite these settings.
 *
 * <p>You may also define the target browser using the {@code SELENIUM_BROWSER}
 * environment variable. If set, this environment variable should be of the
 * form {@code browser[:[version][:platform]]}.
 *
 * @param {(string|webdriver.Browser)} name The name of the target browser;
 *     common defaults are available on the {@link webdriver.Browser} enum.
 * @param {string=} opt_version A desired version; may be omitted if any
 *     version should be used.
 * @param {string=} opt_platform The desired platform; may be omitted if any
 *     version may be used.
 * @return {!Builder} A self reference.
 */
Builder.prototype.forBrowser = function(name, opt_version, opt_platform) {
  this.capabilities_.set(Capability.BROWSER_NAME, name);
  this.capabilities_.set(Capability.VERSION, opt_version || null);
  this.capabilities_.set(Capability.PLATFORM, opt_platform || null);
  return this;
};


/**
 * Sets the proxy configuration to use for WebDriver clients created by this
 * builder. Any calls to {@link #withCapabilities} after this function will
 * overwrite these settings.
 * @param {!webdriver.ProxyConfig} config The configuration to use.
 * @return {!Builder} A self reference.
 */
Builder.prototype.setProxy = function(config) {
  this.capabilities_.setProxy(config);
  return this;
};


/**
 * Sets the logging preferences for the created session. Preferences may be
 * changed by repeated calls, or by calling {@link #withCapabilities}.
 * @param {!(webdriver.logging.Preferences|Object.<string, string>)} prefs The
 *     desired logging preferences.
 * @return {!Builder} A self reference.
 */
Builder.prototype.setLoggingPrefs = function(prefs) {
  this.capabilities_.setLoggingPrefs(prefs);
  return this;
};


/**
 * Sets whether native events should be used.
 * @param {boolean} enabled Whether to enable native events.
 * @return {!Builder} A self reference.
 */
Builder.prototype.setEnableNativeEvents = function(enabled) {
  this.capabilities_.setEnableNativeEvents(enabled);
  return this;
};


/**
 * Sets how elements should be scrolled into view for interaction.
 * @param {number} behavior The desired scroll behavior: either 0 to align with
 *     the top of the viewport or 1 to align with the bottom.
 * @return {!Builder} A self reference.
 */
Builder.prototype.setScrollBehavior = function(behavior) {
  this.capabilities_.setScrollBehavior(behavior);
  return this;
};


/**
 * Sets the default action to take with an unexpected alert before returning
 * an error.
 * @param {string} beahvior The desired behavior; should be "accept", "dismiss",
 *     or "ignore". Defaults to "dismiss".
 * @return {!Builder} A self reference.
 */
Builder.prototype.setAlertBehavior = function(behavior) {
  this.capabilities_.setAlertBehavior(behavior);
  return this;
};


/**
 * Sets Chrome-specific options for drivers created by this builder. Any
 * logging or proxy settings defined on the given options will take precedence
 * over those set through {@link #setLoggingPrefs} and {@link #setProxy},
 * respectively.
 *
 * @param {!chrome.Options} options The ChromeDriver options to use.
 * @return {!Builder} A self reference.
 */
Builder.prototype.setChromeOptions = function(options) {
  this.chromeOptions_ = options;
  return this;
};


/**
 * Sets Firefox-specific options for drivers created by this builder. Any
 * logging or proxy settings defined on the given options will take precedence
 * over those set through {@link #setLoggingPrefs} and {@link #setProxy},
 * respectively.
 *
 * @param {!firefox.Options} options The FirefoxDriver options to use.
 * @return {!Builder} A self reference.
 */
Builder.prototype.setFirefoxOptions = function(options) {
  this.firefoxOptions_ = options;
  return this;
};


/**
 * Sets the control flow that created drivers should execute actions in. If
 * the flow is never set, or is set to {@code null}, it will use the active
 * flow at the time {@link #build()} is called.
 * @param {webdriver.promise.ControlFlow} flow The control flow to use, or
 *     {@code null} to
 * @return {!Builder} A self reference.
 */
Builder.prototype.setControlFlow = function(flow) {
  this.flow_ = flow;
  return this;
};


/**
 * Creates a new WebDriver client based on this builder's current
 * configuration.
 *
 * @return {!webdriver.WebDriver} A new WebDriver instance.
 * @throws {Error} If the current configuration is invalid.
 */
Builder.prototype.build = function() {
  // Create a copy for any changes we may need to make based on the current
  // environment.
  var capabilities = new Capabilities(this.capabilities_);

  var browser = process.env.SELENIUM_BROWSER;
  if (browser) {
    browser = browser.split(/:/, 3);
    capabilities.set(Capability.BROWSER_NAME, browser[0]);
    capabilities.set(Capability.VERSION, browser[1] || null);
    capabilities.set(Capability.PLATFORM, browser[2] || null);
  }

  browser = capabilities.get(Capability.BROWSER_NAME);

  if (typeof browser !== 'string') {
    throw TypeError(
        'Target browser must be a string, but is <' + (typeof browser) + '>;' +
        ' did you forget to call forBrowser()?');
  }

  // Apply browser specific overrides.
  if (browser === Browser.CHROME && this.chromeOptions_) {
    capabilities.merge(this.chromeOptions_.toCapabilities());
  }

  if (browser === Browser.FIREFOX && this.firefoxOptions_) {
    capabilities.merge(this.firefoxOptions_.toCapabilities());
  }

  // Check for a remote browser.
  var url = process.env.SELENIUM_REMOTE_URL || this.url_;
  if (url) {
    var executor = executors.createExecutor(url);
    return WebDriver.createSession(executor, capabilities, this.flow_);
  }

  // Check for a native browser.
  switch (browser) {
    case Browser.CHROME:
      // Requiring 'chrome' above would create a cycle:
      // index -> builder -> chrome -> index
      var chrome = require('./chrome');
      return new chrome.Driver(capabilities, null, this.flow_);

    case Browser.FIREFOX:
      // Requiring 'firefox' above would create a cycle:
      // index -> builder -> firefox -> index
      var firefox = require('./firefox');
      return new firefox.Driver(capabilities, this.flow_);

    case Browser.PHANTOM_JS:
      // Requiring 'phantomjs' would create a cycle:
      // index -> builder -> phantomjs -> index
      var phantomjs = require('./phantomjs');
      return new phantomjs.Driver(capabilities, this.flow_);

    default:
      throw new Error('Do not know how to build driver: ' + browser
          + '; did you forget to call usingServer(url)?');
  }
};


// PUBLIC API


exports.Builder = Builder;
