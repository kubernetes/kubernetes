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

/**
 * @fileoverview Contains several classes for handling commands.
 */

goog.provide('webdriver.Command');
goog.provide('webdriver.CommandExecutor');
goog.provide('webdriver.CommandName');



/**
 * Describes a command to be executed by the WebDriverJS framework.
 * @param {!webdriver.CommandName} name The name of this command.
 * @constructor
 */
webdriver.Command = function(name) {

  /**
   * The name of this command.
   * @private {!webdriver.CommandName}
   */
  this.name_ = name;

  /**
   * The parameters to this command.
   * @private {!Object.<*>}
   */
  this.parameters_ = {};
};


/**
 * @return {!webdriver.CommandName} This command's name.
 */
webdriver.Command.prototype.getName = function() {
  return this.name_;
};


/**
 * Sets a parameter to send with this command.
 * @param {string} name The parameter name.
 * @param {*} value The parameter value.
 * @return {!webdriver.Command} A self reference.
 */
webdriver.Command.prototype.setParameter = function(name, value) {
  this.parameters_[name] = value;
  return this;
};


/**
 * Sets the parameters for this command.
 * @param {!Object.<*>} parameters The command parameters.
 * @return {!webdriver.Command} A self reference.
 */
webdriver.Command.prototype.setParameters = function(parameters) {
  this.parameters_ = parameters;
  return this;
};


/**
 * Returns a named command parameter.
 * @param {string} key The parameter key to look up.
 * @return {*} The parameter value, or undefined if it has not been set.
 */
webdriver.Command.prototype.getParameter = function(key) {
  return this.parameters_[key];
};


/**
 * @return {!Object.<*>} The parameters to send with this command.
 */
webdriver.Command.prototype.getParameters = function() {
  return this.parameters_;
};


/**
 * Enumeration of predefined names command names that all command processors
 * will support.
 * @enum {string}
 */
// TODO: Delete obsolete command names.
webdriver.CommandName = {
  GET_SERVER_STATUS: 'getStatus',

  NEW_SESSION: 'newSession',
  GET_SESSIONS: 'getSessions',
  DESCRIBE_SESSION: 'getSessionCapabilities',

  CLOSE: 'close',
  QUIT: 'quit',

  GET_CURRENT_URL: 'getCurrentUrl',
  GET: 'get',
  GO_BACK: 'goBack',
  GO_FORWARD: 'goForward',
  REFRESH: 'refresh',

  ADD_COOKIE: 'addCookie',
  GET_COOKIE: 'getCookie',
  GET_ALL_COOKIES: 'getCookies',
  DELETE_COOKIE: 'deleteCookie',
  DELETE_ALL_COOKIES: 'deleteAllCookies',

  GET_ACTIVE_ELEMENT: 'getActiveElement',
  FIND_ELEMENT: 'findElement',
  FIND_ELEMENTS: 'findElements',
  FIND_CHILD_ELEMENT: 'findChildElement',
  FIND_CHILD_ELEMENTS: 'findChildElements',

  CLEAR_ELEMENT: 'clearElement',
  CLICK_ELEMENT: 'clickElement',
  SEND_KEYS_TO_ELEMENT: 'sendKeysToElement',
  SUBMIT_ELEMENT: 'submitElement',

  GET_CURRENT_WINDOW_HANDLE: 'getCurrentWindowHandle',
  GET_WINDOW_HANDLES: 'getWindowHandles',
  GET_WINDOW_POSITION: 'getWindowPosition',
  SET_WINDOW_POSITION: 'setWindowPosition',
  GET_WINDOW_SIZE: 'getWindowSize',
  SET_WINDOW_SIZE: 'setWindowSize',
  MAXIMIZE_WINDOW: 'maximizeWindow',

  SWITCH_TO_WINDOW: 'switchToWindow',
  SWITCH_TO_FRAME: 'switchToFrame',
  GET_PAGE_SOURCE: 'getPageSource',
  GET_TITLE: 'getTitle',

  EXECUTE_SCRIPT: 'executeScript',
  EXECUTE_ASYNC_SCRIPT: 'executeAsyncScript',

  GET_ELEMENT_TEXT: 'getElementText',
  GET_ELEMENT_TAG_NAME: 'getElementTagName',
  IS_ELEMENT_SELECTED: 'isElementSelected',
  IS_ELEMENT_ENABLED: 'isElementEnabled',
  IS_ELEMENT_DISPLAYED: 'isElementDisplayed',
  GET_ELEMENT_LOCATION: 'getElementLocation',
  GET_ELEMENT_LOCATION_IN_VIEW: 'getElementLocationOnceScrolledIntoView',
  GET_ELEMENT_SIZE: 'getElementSize',
  GET_ELEMENT_ATTRIBUTE: 'getElementAttribute',
  GET_ELEMENT_VALUE_OF_CSS_PROPERTY: 'getElementValueOfCssProperty',
  ELEMENT_EQUALS: 'elementEquals',

  SCREENSHOT: 'screenshot',
  IMPLICITLY_WAIT: 'implicitlyWait',
  SET_SCRIPT_TIMEOUT: 'setScriptTimeout',
  SET_TIMEOUT: 'setTimeout',

  ACCEPT_ALERT: 'acceptAlert',
  DISMISS_ALERT: 'dismissAlert',
  GET_ALERT_TEXT: 'getAlertText',
  SET_ALERT_TEXT: 'setAlertValue',

  EXECUTE_SQL: 'executeSQL',
  GET_LOCATION: 'getLocation',
  SET_LOCATION: 'setLocation',
  GET_APP_CACHE: 'getAppCache',
  GET_APP_CACHE_STATUS: 'getStatus',
  CLEAR_APP_CACHE: 'clearAppCache',
  IS_BROWSER_ONLINE: 'isBrowserOnline',
  SET_BROWSER_ONLINE: 'setBrowserOnline',

  GET_LOCAL_STORAGE_ITEM: 'getLocalStorageItem',
  GET_LOCAL_STORAGE_KEYS: 'getLocalStorageKeys',
  SET_LOCAL_STORAGE_ITEM: 'setLocalStorageItem',
  REMOVE_LOCAL_STORAGE_ITEM: 'removeLocalStorageItem',
  CLEAR_LOCAL_STORAGE: 'clearLocalStorage',
  GET_LOCAL_STORAGE_SIZE: 'getLocalStorageSize',

  GET_SESSION_STORAGE_ITEM: 'getSessionStorageItem',
  GET_SESSION_STORAGE_KEYS: 'getSessionStorageKey',
  SET_SESSION_STORAGE_ITEM: 'setSessionStorageItem',
  REMOVE_SESSION_STORAGE_ITEM: 'removeSessionStorageItem',
  CLEAR_SESSION_STORAGE: 'clearSessionStorage',
  GET_SESSION_STORAGE_SIZE: 'getSessionStorageSize',

  SET_SCREEN_ORIENTATION: 'setScreenOrientation',
  GET_SCREEN_ORIENTATION: 'getScreenOrientation',

  // These belong to the Advanced user interactions - an element is
  // optional for these commands.
  CLICK: 'mouseClick',
  DOUBLE_CLICK: 'mouseDoubleClick',
  MOUSE_DOWN: 'mouseButtonDown',
  MOUSE_UP: 'mouseButtonUp',
  MOVE_TO: 'mouseMoveTo',
  SEND_KEYS_TO_ACTIVE_ELEMENT: 'sendKeysToActiveElement',

  // These belong to the Advanced Touch API
  TOUCH_SINGLE_TAP: 'touchSingleTap',
  TOUCH_DOWN: 'touchDown',
  TOUCH_UP: 'touchUp',
  TOUCH_MOVE: 'touchMove',
  TOUCH_SCROLL: 'touchScroll',
  TOUCH_DOUBLE_TAP: 'touchDoubleTap',
  TOUCH_LONG_PRESS: 'touchLongPress',
  TOUCH_FLICK: 'touchFlick',

  GET_AVAILABLE_LOG_TYPES: 'getAvailableLogTypes',
  GET_LOG: 'getLog',
  GET_SESSION_LOGS: 'getSessionLogs'
};



/**
 * Handles the execution of {@code webdriver.Command} objects.
 * @interface
 */
webdriver.CommandExecutor = function() {};


/**
 * Executes the given {@code command}. If there is an error executing the
 * command, the provided callback will be invoked with the offending error.
 * Otherwise, the callback will be invoked with a null Error and non-null
 * {@link bot.response.ResponseObject} object.
 * @param {!webdriver.Command} command The command to execute.
 * @param {function(Error, !bot.response.ResponseObject=)} callback the function
 *     to invoke when the command response is ready.
 */
webdriver.CommandExecutor.prototype.execute = goog.abstractMethod;
