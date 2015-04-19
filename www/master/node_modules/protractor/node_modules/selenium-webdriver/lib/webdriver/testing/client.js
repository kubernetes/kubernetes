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

goog.provide('webdriver.testing.Client');

goog.require('goog.json');
goog.require('goog.net.XmlHttp');



/**
 * The client responsible for publishing test events to the server. Each event
 * will be published using a POST {@link goog.net.XmlHttp} request. The body of
 * each request will be a JSON object with the following fields:
 * <ul>
 *   <li>id: An identifier for this client, derived from the window
 *       locations' pathname.
 *   <li>type: The type of event.
 *   <li>data: A JSONObject whose contents will be specific to each event type.
 * </ul>
 *
 * @param {Window=} opt_win The window to pull the path name from for this
 *     client. Defaults to the current window.
 * @param {string=} opt_url The URL to publish test events to. Defaults to
 *     {@link webdriver.testing.Client.DEFAULT_URL}.
 * @constructor
 */
webdriver.testing.Client = function(opt_win, opt_url) {

  /** @private {string} */
  this.id_ = (opt_win || window).location.pathname;

  /** @private {string} */
  this.url_ = opt_url || webdriver.testing.Client.DEFAULT_URL;
};


/**
 * Default URL to publish test events to.
 * @type {string}
 * @const
 */
webdriver.testing.Client.DEFAULT_URL = '/testevent';


/**
 * The types of events that may be published by a TestClient to the server.
 * @enum {string}
 * @private
 */
webdriver.testing.Client.EventType_ = {

  /** Sent to signal that a test suite has been fully initialized. */
  INIT: 'INIT',

  /**
   * Sent when starting a new test.  The data object will have the following
   * fields:
   *  - name: The name of the test.
   */
  START_TEST: 'START_TEST',

  /**
   * Sent when an error has occurred. The data object will have the following
   * fields:
   *  - message: The error message.
   */
  ERROR: 'ERROR',

  /**
   * Sent when all tests have completed. The data object will have the following
   * fields:
   *  - isSuccess: Whether the tests succeeded.
   *  - report: A flat log for the test suite.
   */
  RESULTS: 'RESULTS',

  /**
   * Sent when there is a screenshot for the server to record. The included data
   * object will have two fields:
   *  - name: A debug label for the screenshot.
   *  - data: The PNG screenshot as a base64 string.
   */
  SCREENSHOT: 'SCREENSHOT'
};


/**
 * Sends a simple message to the server, notifying it that the test runner has
 * been initialized.
 */
webdriver.testing.Client.prototype.sendInitEvent = function() {
  this.sendEvent_(webdriver.testing.Client.EventType_.INIT);
};


/**
 * Sends an error event.
 * @param {string} message The error message.
 */
webdriver.testing.Client.prototype.sendErrorEvent = function(message) {
  this.sendEvent_(webdriver.testing.Client.EventType_.ERROR, {
    'message': message
  });
};


/**
 * Sends an event indicating that a new test has started.
 * @param {string} name The name of the test.
 */
webdriver.testing.Client.prototype.sendTestStartedEvent = function(name) {
  this.sendEvent_(webdriver.testing.Client.EventType_.START_TEST, {
    'name': name
  });
};


/**
 * Sends an event to the server indicating that tests have completed.
 * @param {boolean} isSuccess Whether the tests finished successfully.
 * @param {string} report The test log.
 */
webdriver.testing.Client.prototype.sendResultsEvent = function(isSuccess,
    report) {
  this.sendEvent_(webdriver.testing.Client.EventType_.RESULTS, {
    'isSuccess': isSuccess,
    'report': report
  });
};


/**
* Sends a screenshot to be recorded by the server.
* @param {string} data The PNG screenshot as a base64 string.
* @param {string=} opt_name A debug label for the screenshot; if omitted, the
*     server will generate a random name.
*/
webdriver.testing.Client.prototype.sendScreenshotEvent = function(data,
    opt_name) {
  this.sendEvent_(webdriver.testing.Client.EventType_.SCREENSHOT, {
    'name': opt_name,
    'data': data
  });
};


/**
* Sends an event to the server.
* @param {string} type The type of event to send.
* @param {!Object=} opt_data JSON data object to send with the event, if any.
* @private
*/
webdriver.testing.Client.prototype.sendEvent_ = function(type, opt_data) {
  var payload = goog.json.serialize({
    'id': this.id_,
    'type': type,
    'data': opt_data || {}
  });

  var xhr = new goog.net.XmlHttp;
  xhr.open('POST', this.url_, true);
  xhr.send(payload);
  // TODO: Log if the event was not sent properly.
};
