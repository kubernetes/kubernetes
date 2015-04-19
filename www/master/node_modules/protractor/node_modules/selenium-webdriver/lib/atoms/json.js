// Copyright 2012 WebDriver committers
// Copyright 2012 Google Inc.
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
 * @fileoverview Provides JSON utilities that uses native JSON parsing where
 * possible (a feature not currently offered by Closure).
 */

goog.provide('bot.json');

goog.require('bot.userAgent');
goog.require('goog.json');
goog.require('goog.userAgent');


/**
 * @define {boolean} NATIVE_JSON indicates whether the code should rely on the
 * native {@code JSON} functions, if available.
 *
 * <p>The JSON functions can be defined by external libraries like Prototype
 * and setting this flag to false forces the use of Closure's goog.json
 * implementation.
 *
 * <p>If your JavaScript can be loaded by a third_party site and you are wary
 * about relying on the native functions, specify
 * "--define bot.json.NATIVE_JSON=false" to the Closure compiler.
 */
bot.json.NATIVE_JSON = true;


/**
 * Whether the current browser supports the native JSON interface.
 * @const
 * @see http://caniuse.com/#search=JSON
 * @private {boolean}
 */
bot.json.SUPPORTS_NATIVE_JSON_ =
    // List WebKit and Opera first since every supported version of these
    // browsers supports native JSON (and we can compile away large chunks of
    // code for individual fragments by setting the appropriate compiler flags).
    goog.userAgent.WEBKIT || goog.userAgent.OPERA ||
        (goog.userAgent.GECKO && bot.userAgent.isEngineVersion(3.5)) ||
        (goog.userAgent.IE && bot.userAgent.isEngineVersion(8));


/**
 * Converts a JSON object to its string representation.
 * @param {*} jsonObj The input object.
 * @param {?(function(string, *): *)=} opt_replacer A replacer function called
 *     for each (key, value) pair that determines how the value should be
 *     serialized. By default, this just returns the value and allows default
 *     serialization to kick in.
 * @return {string} A JSON string representation of the input object.
 */
bot.json.stringify = bot.json.NATIVE_JSON && bot.json.SUPPORTS_NATIVE_JSON_ ?
    JSON.stringify : goog.json.serialize;


/**
 * Parses a JSON string and returns the result.
 * @param {string} jsonStr The string to parse.
 * @return {*} The JSON object.
 * @throws {Error} If the input string is an invalid JSON string.
 */
bot.json.parse = bot.json.NATIVE_JSON && bot.json.SUPPORTS_NATIVE_JSON_ ?
    JSON.parse : goog.json.parse;
