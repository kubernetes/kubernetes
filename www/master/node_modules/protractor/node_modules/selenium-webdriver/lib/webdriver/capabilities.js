// Copyright 2013 Software Freedom Conservancy
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

/**
 * @fileoverview Defines the webdriver.Capabilities class.
 */

goog.provide('webdriver.Browser');
goog.provide('webdriver.Capabilities');
goog.provide('webdriver.Capability');
goog.provide('webdriver.ProxyConfig');

goog.require('webdriver.logging.Preferences');



/**
 * Recognized browser names.
 * @enum {string}
 */
webdriver.Browser = {
  ANDROID: 'android',
  CHROME: 'chrome',
  FIREFOX: 'firefox',
  INTERNET_EXPLORER: 'internet explorer',
  IPAD: 'iPad',
  IPHONE: 'iPhone',
  OPERA: 'opera',
  PHANTOM_JS: 'phantomjs',
  SAFARI: 'safari',
  HTMLUNIT: 'htmlunit'
};



/**
 * Describes how a proxy should be configured for a WebDriver session.
 * Proxy configuration object, as defined by the WebDriver wire protocol.
 * @typedef {(
 *     {proxyType: string}|
 *     {proxyType: string,
 *      proxyAutoconfigUrl: string}|
 *     {proxyType: string,
 *      ftpProxy: string,
 *      httpProxy: string,
 *      sslProxy: string,
 *      noProxy: string})}
 */
webdriver.ProxyConfig;



/**
 * Common webdriver capability keys.
 * @enum {string}
 */
webdriver.Capability = {

  /**
   * Indicates whether a driver should accept all SSL certs by default. This
   * capability only applies when requesting a new session. To query whether
   * a driver can handle insecure SSL certs, see
   * {@link webdriver.Capability.SECURE_SSL}.
   */
  ACCEPT_SSL_CERTS: 'acceptSslCerts',


  /**
   * The browser name. Common browser names are defined in the
   * {@link webdriver.Browser} enum.
   */
  BROWSER_NAME: 'browserName',

  /**
   * Defines how elements should be scrolled into the viewport for interaction.
   * This capability will be set to zero (0) if elements are aligned with the
   * top of the viewport, or one (1) if aligned with the bottom. The default
   * behavior is to align with the top of the viewport.
   */
  ELEMENT_SCROLL_BEHAVIOR: 'elementScrollBehavior',

  /**
   * Whether the driver is capable of handling modal alerts (e.g. alert,
   * confirm, prompt). To define how a driver <i>should</i> handle alerts,
   * use {@link webdriver.Capability.UNEXPECTED_ALERT_BEHAVIOR}.
   */
  HANDLES_ALERTS: 'handlesAlerts',

  /**
   * Key for the logging driver logging preferences.
   */
  LOGGING_PREFS: 'loggingPrefs',

  /**
   * Whether this session generates native events when simulating user input.
   */
  NATIVE_EVENTS: 'nativeEvents',

  /**
   * Describes the platform the browser is running on. Will be one of
   * ANDROID, IOS, LINUX, MAC, UNIX, or WINDOWS. When <i>requesting</i> a
   * session, ANY may be used to indicate no platform preference (this is
   * semantically equivalent to omitting the platform capability).
   */
  PLATFORM: 'platform',

  /**
   * Describes the proxy configuration to use for a new WebDriver session.
   */
  PROXY: 'proxy',

  /** Whether the driver supports changing the brower's orientation. */
  ROTATABLE: 'rotatable',

  /**
   * Whether a driver is only capable of handling secure SSL certs. To request
   * that a driver accept insecure SSL certs by default, use
   * {@link webdriver.Capability.ACCEPT_SSL_CERTS}.
   */
  SECURE_SSL: 'secureSsl',

  /** Whether the driver supports manipulating the app cache. */
  SUPPORTS_APPLICATION_CACHE: 'applicationCacheEnabled',

  /** Whether the driver supports locating elements with CSS selectors. */
  SUPPORTS_CSS_SELECTORS: 'cssSelectorsEnabled',

  /** Whether the browser supports JavaScript. */
  SUPPORTS_JAVASCRIPT: 'javascriptEnabled',

  /** Whether the driver supports controlling the browser's location info. */
  SUPPORTS_LOCATION_CONTEXT: 'locationContextEnabled',

  /** Whether the driver supports taking screenshots. */
  TAKES_SCREENSHOT: 'takesScreenshot',

  /**
   * Defines how the driver should handle unexpected alerts. The value should
   * be one of "accept", "dismiss", or "ignore.
   */
  UNEXPECTED_ALERT_BEHAVIOR: 'unexpectedAlertBehavior',

  /** Defines the browser version. */
  VERSION: 'version'
};



/**
 * @param {(webdriver.Capabilities|Object)=} opt_other Another set of
 *     capabilities to merge into this instance.
 * @constructor
 */
webdriver.Capabilities = function(opt_other) {

  /** @private {!Object} */
  this.caps_ = {};

  if (opt_other) {
    this.merge(opt_other);
  }
};


/**
 * @return {!webdriver.Capabilities} A basic set of capabilities for Android.
 */
webdriver.Capabilities.android = function() {
  return new webdriver.Capabilities().
      set(webdriver.Capability.BROWSER_NAME, webdriver.Browser.ANDROID).
      set(webdriver.Capability.PLATFORM, 'ANDROID');
};


/**
 * @return {!webdriver.Capabilities} A basic set of capabilities for Chrome.
 */
webdriver.Capabilities.chrome = function() {
  return new webdriver.Capabilities().
      set(webdriver.Capability.BROWSER_NAME, webdriver.Browser.CHROME);
};


/**
 * @return {!webdriver.Capabilities} A basic set of capabilities for Firefox.
 */
webdriver.Capabilities.firefox = function() {
  return new webdriver.Capabilities().
      set(webdriver.Capability.BROWSER_NAME, webdriver.Browser.FIREFOX);
};


/**
 * @return {!webdriver.Capabilities} A basic set of capabilities for
 *     Internet Explorer.
 */
webdriver.Capabilities.ie = function() {
  return new webdriver.Capabilities().
      set(webdriver.Capability.BROWSER_NAME,
          webdriver.Browser.INTERNET_EXPLORER).
      set(webdriver.Capability.PLATFORM, 'WINDOWS');
};


/**
 * @return {!webdriver.Capabilities} A basic set of capabilities for iPad.
 */
webdriver.Capabilities.ipad = function() {
  return new webdriver.Capabilities().
      set(webdriver.Capability.BROWSER_NAME, webdriver.Browser.IPAD).
      set(webdriver.Capability.PLATFORM, 'MAC');
};


/**
 * @return {!webdriver.Capabilities} A basic set of capabilities for iPhone.
 */
webdriver.Capabilities.iphone = function() {
  return new webdriver.Capabilities().
      set(webdriver.Capability.BROWSER_NAME, webdriver.Browser.IPHONE).
      set(webdriver.Capability.PLATFORM, 'MAC');
};


/**
 * @return {!webdriver.Capabilities} A basic set of capabilities for Opera.
 */
webdriver.Capabilities.opera = function() {
  return new webdriver.Capabilities().
      set(webdriver.Capability.BROWSER_NAME, webdriver.Browser.OPERA);
};


/**
 * @return {!webdriver.Capabilities} A basic set of capabilities for
 *     PhantomJS.
 */
webdriver.Capabilities.phantomjs = function() {
  return new webdriver.Capabilities().
      set(webdriver.Capability.BROWSER_NAME, webdriver.Browser.PHANTOM_JS);
};


/**
 * @return {!webdriver.Capabilities} A basic set of capabilities for Safari.
 */
webdriver.Capabilities.safari = function() {
  return new webdriver.Capabilities().
      set(webdriver.Capability.BROWSER_NAME, webdriver.Browser.SAFARI);
};


/**
 * @return {!webdriver.Capabilities} A basic set of capabilities for HTMLUnit.
 */
webdriver.Capabilities.htmlunit = function() {
  return new webdriver.Capabilities().
      set(webdriver.Capability.BROWSER_NAME, webdriver.Browser.HTMLUNIT);
};


/**
 * @return {!webdriver.Capabilities} A basic set of capabilities for HTMLUnit
 *     with enabled Javascript.
 */
webdriver.Capabilities.htmlunitwithjs = function() {
  return new webdriver.Capabilities().
      set(webdriver.Capability.BROWSER_NAME, webdriver.Browser.HTMLUNIT).
      set(webdriver.Capability.SUPPORTS_JAVASCRIPT, true);
};


/** @return {!Object} The JSON representation of this instance. */
webdriver.Capabilities.prototype.toJSON = function() {
  return this.caps_;
};


/**
 * Merges another set of capabilities into this instance. Any duplicates in
 * the provided set will override those already set on this instance.
 * @param {!(webdriver.Capabilities|Object)} other The capabilities to
 *     merge into this instance.
 * @return {!webdriver.Capabilities} A self reference.
 */
webdriver.Capabilities.prototype.merge = function(other) {
  var caps = other instanceof webdriver.Capabilities ?
      other.caps_ : other;
  for (var key in caps) {
    if (caps.hasOwnProperty(key)) {
      this.set(key, caps[key]);
    }
  }
  return this;
};


/**
 * @param {string} key The capability to set.
 * @param {*} value The capability value.  Capability values must be JSON
 *     serializable. Pass {@code null} to unset the capability.
 * @return {!webdriver.Capabilities} A self reference.
 */
webdriver.Capabilities.prototype.set = function(key, value) {
  if (goog.isDefAndNotNull(value)) {
    this.caps_[key] = value;
  } else {
    delete this.caps_[key];
  }
  return this;
};


/**
 * @param {string} key The capability to return.
 * @return {*} The capability with the given key, or {@code null} if it has
 *     not been set.
 */
webdriver.Capabilities.prototype.get = function(key) {
  var val = null;
  if (this.caps_.hasOwnProperty(key)) {
    val = this.caps_[key];
  }
  return goog.isDefAndNotNull(val) ? val : null;
};


/**
 * @param {string} key The capability to check.
 * @return {boolean} Whether the specified capability is set.
 */
webdriver.Capabilities.prototype.has = function(key) {
  return !!this.get(key);
};


/**
 * Sets the logging preferences. Preferences may be specified as a
 * {@link webdriver.logging.Preferences} instance, or a as a map of log-type to
 * log-level.
 * @param {!(webdriver.logging.Preferences|Object.<string, string>)} prefs The
 *     logging preferences.
 * @return {!webdriver.Capabilities} A self reference.
 */
webdriver.Capabilities.prototype.setLoggingPrefs = function(prefs) {
  return this.set(webdriver.Capability.LOGGING_PREFS, prefs);
};


/**
 * Sets the proxy configuration for this instance.
 * @param {webdriver.ProxyConfig} proxy The desired proxy configuration.
 * @return {!webdriver.Capabilities} A self reference.
 */
webdriver.Capabilities.prototype.setProxy = function(proxy) {
  return this.set(webdriver.Capability.PROXY, proxy);
};


/**
 * Sets whether native events should be used.
 * @param {boolean} enabled Whether to enable native events.
 * @return {!webdriver.Capabilities} A self reference.
 */
webdriver.Capabilities.prototype.setEnableNativeEvents = function(enabled) {
  return this.set(webdriver.Capability.NATIVE_EVENTS, enabled);
};


/**
 * Sets how elements should be scrolled into view for interaction.
 * @param {number} behavior The desired scroll behavior: either 0 to align with
 *     the top of the viewport or 1 to align with the bottom.
 * @return {!webdriver.Capabilities} A self reference.
 */
webdriver.Capabilities.prototype.setScrollBehavior = function(behavior) {
  return this.set(webdriver.Capability.ELEMENT_SCROLL_BEHAVIOR, behavior);
};


/**
 * Sets the default action to take with an unexpected alert before returning
 * an error.
 * @param {string} behavior The desired behavior; should be "accept", "dismiss",
 *     or "ignore". Defaults to "dismiss".
 * @return {!webdriver.Capabilities} A self reference.
 */
webdriver.Capabilities.prototype.setAlertBehavior = function(behavior) {
  return this.set(webdriver.Capability.UNEXPECTED_ALERT_BEHAVIOR, behavior);
};
