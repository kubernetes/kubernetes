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

goog.provide('webdriver.FirefoxDomExecutor');

goog.require('bot.response');
goog.require('goog.json');
goog.require('goog.userAgent.product');
goog.require('webdriver.Command');
goog.require('webdriver.CommandName');



/**
 * @constructor
 * @implements {webdriver.CommandExecutor}
 */
webdriver.FirefoxDomExecutor = function() {
  if (!webdriver.FirefoxDomExecutor.isAvailable()) {
    throw Error(
        'The current environment does not support the FirefoxDomExecutor');
  }

  /** @private {!Document} */
  this.doc_ = document;

  /** @private {!Element} */
  this.docElement_ = document.documentElement;

  this.docElement_.addEventListener(
      webdriver.FirefoxDomExecutor.EventType_.RESPONSE,
      goog.bind(this.onResponse_, this), false);
};


/**
 * @return {boolean} Whether the current environment supports the
 *     FirefoxDomExecutor.
 */
webdriver.FirefoxDomExecutor.isAvailable = function() {
  return goog.userAgent.product.FIREFOX &&
      typeof document !== 'undefined' &&
      document.documentElement &&
      goog.isFunction(document.documentElement.hasAttribute) &&
      document.documentElement.hasAttribute('webdriver');
};


/**
 * Attributes used to communicate with the FirefoxDriver extension.
 * @enum {string}
 * @private
 */
webdriver.FirefoxDomExecutor.Attribute_ = {
  COMMAND: 'command',
  RESPONSE: 'response'
};


/**
 * Events used to communicate with the FirefoxDriver extension.
 * @enum {string}
 * @private
 */
webdriver.FirefoxDomExecutor.EventType_ = {
  COMMAND: 'webdriverCommand',
  RESPONSE: 'webdriverResponse'
};


/**
 * The pending command, if any.
 * @private {?{name:string, callback:!Function}}
 */
webdriver.FirefoxDomExecutor.prototype.pendingCommand_ = null;


/** @override */
webdriver.FirefoxDomExecutor.prototype.execute = function(command, callback) {
  if (this.pendingCommand_) {
    throw Error('Currently awaiting a command response!');
  }

  this.pendingCommand_ = {
    name: command.getName(),
    callback: callback
  };

  var parameters = command.getParameters();

  // There are two means for communicating with the FirefoxDriver: via
  // HTTP using WebDriver's wire protocol and over the DOM using a custom
  // JSON protocol. This class uses the latter. When the FirefoxDriver receives
  // commands over HTTP, it builds a parameters object from the URL parameters.
  // When an element ID is sent in the URL, it'll be decoded as just id:string
  // instead of id:{ELEMENT:string}. When switching to a frame by element,
  // however, the element ID is not sent through the URL, so we must make sure
  // to encode that parameter properly here. It would be nice if we unified
  // the two protocols used by the FirefoxDriver...
  if (parameters['id'] &&
      parameters['id']['ELEMENT'] &&
      command.getName() != webdriver.CommandName.SWITCH_TO_FRAME) {
    parameters['id'] = parameters['id']['ELEMENT'];
  }
  var json = goog.json.serialize({
    'name': command.getName(),
    'sessionId': parameters['sessionId'],
    'parameters': parameters
  });
  this.docElement_.setAttribute(
      webdriver.FirefoxDomExecutor.Attribute_.COMMAND, json);

  var event = this.doc_.createEvent('Event');
  event.initEvent(webdriver.FirefoxDomExecutor.EventType_.COMMAND,
      /*canBubble=*/true, /*cancelable=*/true);

  this.docElement_.dispatchEvent(event);
};


/** @private */
webdriver.FirefoxDomExecutor.prototype.onResponse_ = function() {
  if (!this.pendingCommand_) {
    return;  // Not expecting a response.
  }

  var command = this.pendingCommand_;
  this.pendingCommand_ = null;

  var json = this.docElement_.getAttribute(
      webdriver.FirefoxDomExecutor.Attribute_.RESPONSE);
  if (!json) {
    command.callback(Error('Empty command response!'));
    return;
  }

  this.docElement_.removeAttribute(
      webdriver.FirefoxDomExecutor.Attribute_.COMMAND);
  this.docElement_.removeAttribute(
      webdriver.FirefoxDomExecutor.Attribute_.RESPONSE);

  try {
    var response = bot.response.checkResponse(
        /** @type {!bot.response.ResponseObject} */ (goog.json.parse(json)));
  } catch (ex) {
    command.callback(ex);
    return;
  }

  // Prior to Selenium 2.35.0, two commands are required to fully create a
  // session: one to allocate the session, and another to fetch the
  // capabilities.
  if (command.name == webdriver.CommandName.NEW_SESSION &&
      goog.isString(response['value'])) {
    var cmd = new webdriver.Command(webdriver.CommandName.DESCRIBE_SESSION).
        setParameter('sessionId', response['value']);
    this.execute(cmd, command.callback);
  } else {
    command.callback(null, response);
  }
};
