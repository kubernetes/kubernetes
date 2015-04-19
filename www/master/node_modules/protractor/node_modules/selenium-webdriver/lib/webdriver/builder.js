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

goog.provide('webdriver.Builder');

goog.require('goog.userAgent');
goog.require('webdriver.AbstractBuilder');
goog.require('webdriver.FirefoxDomExecutor');
goog.require('webdriver.WebDriver');
goog.require('webdriver.http.CorsClient');
goog.require('webdriver.http.Executor');
goog.require('webdriver.http.XhrClient');
goog.require('webdriver.process');



/**
 * @constructor
 * @extends {webdriver.AbstractBuilder}
 */
webdriver.Builder = function() {
  goog.base(this);

  /**
   * ID of an existing WebDriver session that new clients should use.
   * Initialized from the value of the
   * {@link webdriver.AbstractBuilder.SESSION_ID_ENV} environment variable, but
   * may be overridden using
   * {@link webdriver.AbstractBuilder#usingSession}.
   * @private {string}
   */
  this.sessionId_ =
      webdriver.process.getEnv(webdriver.Builder.SESSION_ID_ENV);
};
goog.inherits(webdriver.Builder, webdriver.AbstractBuilder);


/**
 * Environment variable that defines the session ID of an existing WebDriver
 * session to use when creating clients. If set, all new Builder instances will
 * default to creating clients that use this session. To create a new session,
 * use {@code #useExistingSession(boolean)}. The use of this environment
 * variable requires that {@link webdriver.AbstractBuilder.SERVER_URL_ENV} also
 * be set.
 * @type {string}
 * @const
 * @see webdriver.process.getEnv
 */
webdriver.Builder.SESSION_ID_ENV = 'wdsid';


/**
 * Configures the builder to create a client that will use an existing WebDriver
 * session.
 * @param {string} id The existing session ID to use.
 * @return {!webdriver.AbstractBuilder} This Builder instance for chain calling.
 */
webdriver.Builder.prototype.usingSession = function(id) {
  this.sessionId_ = id;
  return this;
};


/**
 * @return {string} The ID of the session, if any, this builder is configured
 *     to reuse.
 */
webdriver.Builder.prototype.getSession = function() {
  return this.sessionId_;
};


/**
 * @override
 */
webdriver.Builder.prototype.build = function() {
  if (goog.userAgent.GECKO && document.readyState != 'complete') {
    throw Error('Cannot create driver instance before window.onload');
  }

  var executor;

  if (webdriver.FirefoxDomExecutor.isAvailable()) {
    executor = new webdriver.FirefoxDomExecutor();
    return webdriver.WebDriver.createSession(executor, this.getCapabilities());
  } else {
    var url = this.getServerUrl() ||
        webdriver.AbstractBuilder.DEFAULT_SERVER_URL;
    var client;
    if (url[0] == '/') {
      var origin = window.location.origin ||
          (window.location.protocol + '//' + window.location.host);
      client = new webdriver.http.XhrClient(origin + url);
    } else {
      client = new webdriver.http.CorsClient(url);
    }
    executor = new webdriver.http.Executor(client);

    if (this.getSession()) {
      return webdriver.WebDriver.attachToSession(executor, this.getSession());
    } else {
      throw new Error('Unable to create a new client for this browser. The ' +
          'WebDriver session ID has not been defined.');
    }
  }
};
