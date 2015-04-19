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

/** @fileoverview A XHR client. */

goog.provide('webdriver.http.XhrClient');

goog.require('goog.json');
goog.require('goog.net.XmlHttp');
goog.require('webdriver.http.Response');



/**
 * A HTTP client that sends requests using XMLHttpRequests.
 * @param {string} url URL for the WebDriver server to send commands to.
 * @constructor
 * @implements {webdriver.http.Client}
 */
webdriver.http.XhrClient = function(url) {

  /** @private {string} */
  this.url_ = url;
};


/** @override */
webdriver.http.XhrClient.prototype.send = function(request, callback) {
  try {
    var xhr = /** @type {!XMLHttpRequest} */ (goog.net.XmlHttp());
    var url = this.url_ + request.path;
    xhr.open(request.method, url, true);

    xhr.onload = function() {
      callback(null, webdriver.http.Response.fromXmlHttpRequest(xhr));
    };

    xhr.onerror = function() {
      callback(Error([
        'Unable to send request: ', request.method, ' ', url,
        '\nOriginal request:\n', request
      ].join('')));
    };

    for (var header in request.headers) {
      xhr.setRequestHeader(header, request.headers[header] + '');
    }

    xhr.send(goog.json.serialize(request.data));
  } catch (ex) {
    callback(ex);
  }
};
