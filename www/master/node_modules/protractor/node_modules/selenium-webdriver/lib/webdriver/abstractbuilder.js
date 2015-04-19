// Copyright 2012 Software Freedom Conservancy. All Rights Reserved.
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

goog.provide('webdriver.AbstractBuilder');

goog.require('webdriver.Capabilities');
goog.require('webdriver.process');



/**
 * Creates new {@code webdriver.WebDriver} clients.  Upon instantiation, each
 * Builder will configure itself based on the following environment variables:
 * <dl>
 *   <dt>{@code webdriver.AbstractBuilder.SERVER_URL_ENV}</dt>
 *   <dd>Defines the remote WebDriver server that should be used for command
 *       command execution; may be overridden using
 *       {@code webdriver.AbstractBuilder.prototype.usingServer}.</dd>
 * </dl>
 * @constructor
 */
webdriver.AbstractBuilder = function() {

  /**
   * URL of the remote server to use for new clients; initialized from the
   * value of the {@link webdriver.AbstractBuilder.SERVER_URL_ENV} environment
   * variable, but may be overridden using
   * {@link webdriver.AbstractBuilder#usingServer}.
   * @private {string}
   */
  this.serverUrl_ = webdriver.process.getEnv(
      webdriver.AbstractBuilder.SERVER_URL_ENV);

  /**
   * The desired capabilities to use when creating a new session.
   * @private {!webdriver.Capabilities}
   */
  this.capabilities_ = new webdriver.Capabilities();
};


/**
 * Environment variable that defines the URL of the WebDriver server that
 * should be used for all new WebDriver clients. This setting may be overridden
 * using {@code #usingServer(url)}.
 * @type {string}
 * @const
 * @see webdriver.process.getEnv
 */
webdriver.AbstractBuilder.SERVER_URL_ENV = 'wdurl';


/**
 * The default URL of the WebDriver server to use if
 * {@link webdriver.AbstractBuilder.SERVER_URL_ENV} is not set.
 * @type {string}
 * @const
 */
webdriver.AbstractBuilder.DEFAULT_SERVER_URL = 'http://localhost:4444/wd/hub';


/**
 * Configures which WebDriver server should be used for new sessions. Overrides
 * the value loaded from the {@link webdriver.AbstractBuilder.SERVER_URL_ENV}
 * upon creation of this instance.
 * @param {string} url URL of the server to use.
 * @return {!webdriver.AbstractBuilder} This Builder instance for chain calling.
 */
webdriver.AbstractBuilder.prototype.usingServer = function(url) {
  this.serverUrl_ = url;
  return this;
};


/**
 * @return {string} The URL of the WebDriver server this instance is configured
 *     to use.
 */
webdriver.AbstractBuilder.prototype.getServerUrl = function() {
  return this.serverUrl_;
};


/**
 * Sets the desired capabilities when requesting a new session. This will
 * overwrite any previously set desired capabilities.
 * @param {!(Object|webdriver.Capabilities)} capabilities The desired
 *     capabilities for a new session.
 * @return {!webdriver.AbstractBuilder} This Builder instance for chain calling.
 */
webdriver.AbstractBuilder.prototype.withCapabilities = function(capabilities) {
  this.capabilities_ = new webdriver.Capabilities(capabilities);
  return this;
};


/**
 * @return {!webdriver.Capabilities} The current desired capabilities for this
 *     builder.
 */
webdriver.AbstractBuilder.prototype.getCapabilities = function() {
  return this.capabilities_;
};


/**
 * Sets the logging preferences for the created session. Preferences may be
 * changed by repeated calls, or by calling {@link #withCapabilities}.
 * @param {!(webdriver.logging.Preferences|Object.<string, string>)} prefs The
 *     desired logging preferences.
 * @return {!webdriver.AbstractBuilder} This Builder instance for chain calling.
 */
webdriver.AbstractBuilder.prototype.setLoggingPreferences = function(prefs) {
  this.capabilities_.set(webdriver.Capability.LOGGING_PREFS, prefs);
  return this;
};


/**
 * Builds a new {@link webdriver.WebDriver} instance using this builder's
 * current configuration.
 * @return {!webdriver.WebDriver} A new WebDriver client.
 */
webdriver.AbstractBuilder.prototype.build = goog.abstractMethod;
