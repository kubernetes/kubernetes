// Copyright 2010 WebDriver committers
// Copyright 2010 Google Inc.
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
 * @fileoverview Utilities for working with errors as defined by WebDriver's
 * wire protocol: http://code.google.com/p/selenium/wiki/JsonWireProtocol.
 */

goog.provide('bot.Error');
goog.provide('bot.ErrorCode');


/**
 * Error codes from the WebDriver wire protocol:
 * http://code.google.com/p/selenium/wiki/JsonWireProtocol#Response_Status_Codes
 *
 * @enum {number}
 */
bot.ErrorCode = {
  SUCCESS: 0,  // Included for completeness

  NO_SUCH_ELEMENT: 7,
  NO_SUCH_FRAME: 8,
  UNKNOWN_COMMAND: 9,
  UNSUPPORTED_OPERATION: 9,  // Alias.
  STALE_ELEMENT_REFERENCE: 10,
  ELEMENT_NOT_VISIBLE: 11,
  INVALID_ELEMENT_STATE: 12,
  UNKNOWN_ERROR: 13,
  ELEMENT_NOT_SELECTABLE: 15,
  JAVASCRIPT_ERROR: 17,
  XPATH_LOOKUP_ERROR: 19,
  TIMEOUT: 21,
  NO_SUCH_WINDOW: 23,
  INVALID_COOKIE_DOMAIN: 24,
  UNABLE_TO_SET_COOKIE: 25,
  /** @deprecated */
  MODAL_DIALOG_OPENED: 26,
  UNEXPECTED_ALERT_OPEN: 26,
  NO_SUCH_ALERT: 27,
  /** @deprecated */
  NO_MODAL_DIALOG_OPEN: 27,
  SCRIPT_TIMEOUT: 28,
  INVALID_ELEMENT_COORDINATES: 29,
  IME_NOT_AVAILABLE: 30,
  IME_ENGINE_ACTIVATION_FAILED: 31,
  INVALID_SELECTOR_ERROR: 32,
  SESSION_NOT_CREATED: 33,
  MOVE_TARGET_OUT_OF_BOUNDS: 34,
  SQL_DATABASE_ERROR: 35,
  INVALID_XPATH_SELECTOR: 51,
  INVALID_XPATH_SELECTOR_RETURN_TYPE: 52,
  // The following error codes are derived straight from HTTP return codes.
  METHOD_NOT_ALLOWED: 405
};



/**
 * Error extension that includes error status codes from the WebDriver wire
 * protocol:
 * http://code.google.com/p/selenium/wiki/JsonWireProtocol#Response_Status_Codes
 *
 * @param {!bot.ErrorCode} code The error's status code.
 * @param {string=} opt_message Optional error message.
 * @constructor
 * @extends {Error}
 */
bot.Error = function(code, opt_message) {

  /**
   * This error's status code.
   * @type {!bot.ErrorCode}
   */
  this.code = code;

  /** @type {string} */
  this.state =
      bot.Error.CODE_TO_STATE_[code] || bot.Error.State.UNKNOWN_ERROR;

  /** @override */
  this.message = opt_message || '';

  var name = this.state.replace(/((?:^|\s+)[a-z])/g, function(str) {
    // IE<9 does not support String#trim(). Also, IE does not include 0xa0
    // (the non-breaking-space) in the \s character class, so we have to
    // explicitly include it.
    return str.toUpperCase().replace(/^[\s\xa0]+/g, '');
  });

  var l = name.length - 'Error'.length;
  if (l < 0 || name.indexOf('Error', l) != l) {
    name += 'Error';
  }

  /** @override */
  this.name = name;

  // Generate a stacktrace for our custom error; ensure the error has our
  // custom name and message so the stack prints correctly in all browsers.
  var template = new Error(this.message);
  template.name = this.name;

  /** @override */
  this.stack = template.stack || '';
};
goog.inherits(bot.Error, Error);


/**
 * Status strings enumerated in the W3C WebDriver working draft.
 * @enum {string}
 * @see http://www.w3.org/TR/webdriver/#status-codes
 */
bot.Error.State = {
  ELEMENT_NOT_SELECTABLE: 'element not selectable',
  ELEMENT_NOT_VISIBLE: 'element not visible',
  IME_ENGINE_ACTIVATION_FAILED: 'ime engine activation failed',
  IME_NOT_AVAILABLE: 'ime not available',
  INVALID_COOKIE_DOMAIN: 'invalid cookie domain',
  INVALID_ELEMENT_COORDINATES: 'invalid element coordinates',
  INVALID_ELEMENT_STATE: 'invalid element state',
  INVALID_SELECTOR: 'invalid selector',
  JAVASCRIPT_ERROR: 'javascript error',
  MOVE_TARGET_OUT_OF_BOUNDS: 'move target out of bounds',
  NO_SUCH_ALERT: 'no such alert',
  NO_SUCH_DOM: 'no such dom',
  NO_SUCH_ELEMENT: 'no such element',
  NO_SUCH_FRAME: 'no such frame',
  NO_SUCH_WINDOW: 'no such window',
  SCRIPT_TIMEOUT: 'script timeout',
  SESSION_NOT_CREATED: 'session not created',
  STALE_ELEMENT_REFERENCE: 'stale element reference',
  SUCCESS: 'success',
  TIMEOUT: 'timeout',
  UNABLE_TO_SET_COOKIE: 'unable to set cookie',
  UNEXPECTED_ALERT_OPEN: 'unexpected alert open',
  UNKNOWN_COMMAND: 'unknown command',
  UNKNOWN_ERROR: 'unknown error',
  UNSUPPORTED_OPERATION: 'unsupported operation'
};


/**
 * A map of error codes to state string.
 * @private {!Object.<bot.ErrorCode, bot.Error.State>}
 */
bot.Error.CODE_TO_STATE_ = {};
goog.scope(function() {
  var map = bot.Error.CODE_TO_STATE_;
  var code = bot.ErrorCode;
  var state = bot.Error.State;

  map[code.ELEMENT_NOT_SELECTABLE] = state.ELEMENT_NOT_SELECTABLE;
  map[code.ELEMENT_NOT_VISIBLE] = state.ELEMENT_NOT_VISIBLE;
  map[code.IME_ENGINE_ACTIVATION_FAILED] = state.IME_ENGINE_ACTIVATION_FAILED;
  map[code.IME_NOT_AVAILABLE] = state.IME_NOT_AVAILABLE;
  map[code.INVALID_COOKIE_DOMAIN] = state.INVALID_COOKIE_DOMAIN;
  map[code.INVALID_ELEMENT_COORDINATES] = state.INVALID_ELEMENT_COORDINATES;
  map[code.INVALID_ELEMENT_STATE] = state.INVALID_ELEMENT_STATE;
  map[code.INVALID_SELECTOR_ERROR] = state.INVALID_SELECTOR;
  map[code.INVALID_XPATH_SELECTOR] = state.INVALID_SELECTOR;
  map[code.INVALID_XPATH_SELECTOR_RETURN_TYPE] = state.INVALID_SELECTOR;
  map[code.JAVASCRIPT_ERROR] = state.JAVASCRIPT_ERROR;
  map[code.METHOD_NOT_ALLOWED] = state.UNSUPPORTED_OPERATION;
  map[code.MOVE_TARGET_OUT_OF_BOUNDS] = state.MOVE_TARGET_OUT_OF_BOUNDS;
  map[code.NO_MODAL_DIALOG_OPEN] = state.NO_SUCH_ALERT;
  map[code.NO_SUCH_ALERT] = state.NO_SUCH_ALERT;
  map[code.NO_SUCH_ELEMENT] = state.NO_SUCH_ELEMENT;
  map[code.NO_SUCH_FRAME] = state.NO_SUCH_FRAME;
  map[code.NO_SUCH_WINDOW] = state.NO_SUCH_WINDOW;
  map[code.SCRIPT_TIMEOUT] = state.SCRIPT_TIMEOUT;
  map[code.SESSION_NOT_CREATED] = state.SESSION_NOT_CREATED;
  map[code.STALE_ELEMENT_REFERENCE] = state.STALE_ELEMENT_REFERENCE;
  map[code.SUCCESS] = state.SUCCESS;
  map[code.TIMEOUT] = state.TIMEOUT;
  map[code.UNABLE_TO_SET_COOKIE] = state.UNABLE_TO_SET_COOKIE;
  map[code.MODAL_DIALOG_OPENED] = state.UNEXPECTED_ALERT_OPEN;
  map[code.UNEXPECTED_ALERT_OPEN] = state.UNEXPECTED_ALERT_OPEN
  map[code.UNKNOWN_ERROR] = state.UNKNOWN_ERROR;
  map[code.UNSUPPORTED_OPERATION] = state.UNKNOWN_COMMAND;
});  // goog.scope


/**
 * Flag used for duck-typing when this code is embedded in a Firefox extension.
 * This is required since an Error thrown in one component and then reported
 * to another will fail instanceof checks in the second component.
 * @type {boolean}
 */
bot.Error.prototype.isAutomationError = true;


if (goog.DEBUG) {
  /** @return {string} The string representation of this error. */
  bot.Error.prototype.toString = function() {
    return this.name + ': ' + this.message;
  };
}
