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

goog.provide('webdriver.http.CorsClient');

goog.require('goog.json');
goog.require('webdriver.http.Response');



/**
 * Communicates with a WebDriver server, which may be on a different domain,
 * using the <a href="http://www.w3.org/TR/cors/">cross-origin resource sharing
 * </a> (CORS) extension to WebDriver's JSON wire protocol.
 *
 * <p>Each command from the standard JSON protocol will be encoded in a
 * JSON object with the following form:
 * {method:string, path:string, data:!Object}
 *
 * <p>The encoded command is then sent as a POST request to the server's /xdrpc
 * endpoint.  The server will decode the command, re-route it to the appropriate
 * handler, and then return the command's response as a standard JSON response
 * object.  The JSON responses will <em>always</em> be returned with a 200
 * response from the server; clients must rely on the response's "status" field
 * to determine whether the command succeeded.
 *
 * <p>This client cannot be used with the standard wire protocol due to
 * limitations in the various browser implementations of the CORS specification:
 * <ul>
 *   <li>IE's <a href="http://goo.gl/6l3kA">XDomainRequest</a> object is only
 *     capable of generating the types of requests that may be generated through
 *     a standard <a href="http://goo.gl/vgzAU">HTML form</a> - it can not send
 *     DELETE requests, as is required in the wire protocol.
 *   <li>WebKit's implementation of CORS does not follow the spec and forbids
 *     redirects: https://bugs.webkit.org/show_bug.cgi?id=57600
 *     This limitation appears to be intentional and is documented in WebKit's
 *     Layout tests:
 *     //LayoutTests/http/tests/xmlhttprequest/access-control-and-redirects.html
 *   <li>If the server does not return a 2xx response, IE and Opera's
 *     implementations will fire the XDomainRequest/XMLHttpRequest object's
 *     onerror handler, but without the corresponding response text returned by
 *     the server. This renders IE and Opera incapable of handling command
 *     failures in the standard JSON protocol.
 * </ul>
 *
 * @param {string} url URL for the WebDriver server to send commands to.
 * @constructor
 * @implements {webdriver.http.Client}
 * @see <a href="http://www.w3.org/TR/cors/">CORS Spec</a>
 * @see <a href="http://code.google.com/p/selenium/wiki/JsonWireProtocol">
 *     JSON wire protocol</a>
 */
webdriver.http.CorsClient = function(url) {
  if (!webdriver.http.CorsClient.isAvailable()) {
    throw Error('The current environment does not support cross-origin ' +
        'resource sharing');
  }

  /** @private {string} */
  this.url_ = url + webdriver.http.CorsClient.XDRPC_ENDPOINT;
};


/**
 * Resource URL to send commands to on the server.
 * @type {string}
 * @const
 */
webdriver.http.CorsClient.XDRPC_ENDPOINT = '/xdrpc';


/**
 * Tests whether the current environment supports cross-origin resource sharing.
 * @return {boolean} Whether cross-origin resource sharing is supported.
 * @see http://www.w3.org/TR/cors/
 */
webdriver.http.CorsClient.isAvailable = function() {
  return typeof XDomainRequest !== 'undefined' ||
      (typeof XMLHttpRequest !== 'undefined' &&
          goog.isBoolean(new XMLHttpRequest().withCredentials));
};


/** @override */
webdriver.http.CorsClient.prototype.send = function(request, callback) {
  try {
    var xhr = new (typeof XDomainRequest !== 'undefined' ?
        XDomainRequest : XMLHttpRequest);
    xhr.open('POST', this.url_, true);

    xhr.onload = function() {
      callback(null, webdriver.http.Response.fromXmlHttpRequest(
          /** @type {!XMLHttpRequest} */ (xhr)));
    };

    var url = this.url_;
    xhr.onerror = function() {
      callback(Error([
        'Unable to send request: POST ', url,
        '\nPerhaps the server did not respond to the preflight request ',
        'with valid access control headers?'
      ].join('')));
    };

    // Define event handlers for all events on the XDomainRequest. Apparently,
    // if we don't do this, IE9+10 will silently abort our request. Yay IE.
    // Note, we're not using goog.nullFunction, because it tends to get
    // optimized away by the compiler, which leaves us where we were before.
    xhr.onprogress = xhr.ontimeout = function() {};

    xhr.send(goog.json.serialize({
      'method': request.method,
      'path': request.path,
      'data': request.data
    }));
  } catch (ex) {
    callback(ex);
  }
};
