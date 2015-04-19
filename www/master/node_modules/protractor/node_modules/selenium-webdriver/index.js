// Copyright 2012 Selenium committers
// Copyright 2012 Software Freedom Conservancy
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//     You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview The main user facing module. Exports WebDriver's primary
 * public API and provides convenience assessors to certain sub-modules.
 */

var base = require('./_base');
var builder = require('./builder');
var error = require('./error');


// NOTE: the remainder of this file is nasty and verbose, but the annotations
// are necessary to guide the Closure Compiler's type analysis. Without them,
// we would not be able to extract any meaningful API documentation.


/** @type {function(new: webdriver.ActionSequence)} */
exports.ActionSequence = base.require('webdriver.ActionSequence');


/** @type {function(new: builder.Builder)} */
exports.Builder = builder.Builder;


/** @type {webdriver.By.} */
exports.By = base.require('webdriver.By');


/** @type {function(new: webdriver.Capabilities)} */
exports.Capabilities = base.require('webdriver.Capabilities');


/** @type {function(new: webdriver.Command)} */
exports.Command = base.require('webdriver.Command');


/** @type {function(new: webdriver.EventEmitter)} */
exports.EventEmitter = base.require('webdriver.EventEmitter');


/** @type {function(new: webdriver.Session)} */
exports.Session = base.require('webdriver.Session');


/** @type {function(new: webdriver.WebDriver)} */
exports.WebDriver = base.require('webdriver.WebDriver');


/** @type {function(new: webdriver.WebElement)} */
exports.WebElement = base.require('webdriver.WebElement');


/** @type {function(new: webdriver.WebElementPromise)} */
exports.WebElementPromise = base.require('webdriver.WebElementPromise');


// Export the remainder of our API through getters to keep things cleaner
// when this module is used in a REPL environment.


/** @type {webdriver.Browser.} */
(exports.__defineGetter__('Browser', function() {
  return base.require('webdriver.Browser');
}));


/** @type {webdriver.Button.} */
(exports.__defineGetter__('Button', function() {
  return base.require('webdriver.Button');
}));


/** @type {webdriver.Capability.} */
(exports.__defineGetter__('Capability', function() {
  return base.require('webdriver.Capability');
}));


/** @type {webdriver.CommandName.} */
(exports.__defineGetter__('CommandName', function() {
  return base.require('webdriver.CommandName');
}));


/** @type {webdriver.Key.} */
(exports.__defineGetter__('Key', function() {
  return base.require('webdriver.Key');
}));


/** @type {error.} */
(exports.__defineGetter__('error', function() {
  return error;
}));


/** @type {error.} */
(exports.__defineGetter__('error', function() {
  return error;
}));


/** @type {webdriver.logging.} */
(exports.__defineGetter__('logging', function() {
  return base.exportPublicApi('webdriver.logging');
}));


/** @type {webdriver.promise.} */
(exports.__defineGetter__('promise', function() {
  return base.exportPublicApi('webdriver.promise');
}));


/** @type {webdriver.stacktrace.} */
(exports.__defineGetter__('stacktrace', function() {
  return base.exportPublicApi('webdriver.stacktrace');
}));


/** @type {webdriver.until.} */
(exports.__defineGetter__('until', function() {
  return base.exportPublicApi('webdriver.until');
}));
