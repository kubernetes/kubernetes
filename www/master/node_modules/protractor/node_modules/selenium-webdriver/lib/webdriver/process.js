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
 * @fileoverview Provides access to the current process' environment variables.
 * When running in node, this is simply a wrapper for {@code process.env}.
 * When running in a browser, environment variables are loaded by parsing the
 * current URL's query string. Variables that have more than one variable will
 * be initialized to the JSON representation of the array of all values,
 * otherwise the variable will be initialized to a sole string value. If a
 * variable does not have any values, but is nonetheless present in the query
 * string, it will be initialized to an empty string.
 * After the initial parsing, environment variables must be queried and set
 * through the API defined in this file.
 */

goog.provide('webdriver.process');

goog.require('goog.Uri');
goog.require('goog.array');
goog.require('goog.json');


/**
 * @return {boolean} Whether the current process is Node's native process
 *     object.
 */
webdriver.process.isNative = function() {
  return webdriver.process.IS_NATIVE_PROCESS_;
};


/**
 * Queries for a named environment variable.
 * @param {string} name The name of the environment variable to look up.
 * @param {string=} opt_default The default value if the named variable is not
 *     defined.
 * @return {string} The queried environment variable.
 */
webdriver.process.getEnv = function(name, opt_default) {
  var value = webdriver.process.PROCESS_.env[name];
  return goog.isDefAndNotNull(value) ? value : opt_default;
};


/**
 * Sets an environment value. If the new value is either null or undefined, the
 *     environment variable will be cleared.
 * @param {string} name The value to set.
 * @param {*} value The new value; will be coerced to a string.
 */
webdriver.process.setEnv = function(name, value) {
  webdriver.process.PROCESS_.env[name] =
      goog.isDefAndNotNull(value) ? value + '' : null;
};


/**
 * Whether the current environment is using Node's native process object.
 * @private {boolean}
 * @const
 */
webdriver.process.IS_NATIVE_PROCESS_ = typeof process !== 'undefined';


/**
 * Initializes a process object for use in a browser window.
 * @param {!Window=} opt_window The window object to initialize the process
 *     from; if not specified, will default to the current window. Should only
 *     be set for unit testing.
 * @return {!Object} The new process object.
 * @private
 */
webdriver.process.initBrowserProcess_ = function(opt_window) {
  var process = {'env': {}};

  var win = opt_window;
  if (!win && typeof window != 'undefined') {
    win = window;
  }

  // Initialize the global error handler.
  if (win) {
    // Initialize the environment variable map by parsing the current URL query
    // string.
    if (win.location) {
      var data = new goog.Uri(win.location).getQueryData();
      goog.array.forEach(data.getKeys(), function(key) {
        var values = data.getValues(key);
        process.env[key] = values.length == 0 ? '' :
                           values.length == 1 ? values[0] :
                           goog.json.serialize(values);
      });
    }
  }

  return process;
};


/**
 * The global process object to use. Will either be Node's global
 * {@code process} object, or an approximation of it for use in a browser
 * environment.
 * @private {!Object}
 * @const
 */
webdriver.process.PROCESS_ = webdriver.process.IS_NATIVE_PROCESS_ ? process :
    webdriver.process.initBrowserProcess_();
