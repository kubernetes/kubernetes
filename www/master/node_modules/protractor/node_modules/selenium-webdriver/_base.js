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
 * @fileoverview The base module responsible for bootstrapping the Closure
 * library and providing a means of loading Closure-based modules.
 *
 * <p>Each script loaded by this module will be granted access to this module's
 * {@code require} function; all required non-native modules must be specified
 * relative to <em>this</em> module.
 *
 * <p>This module will load all scripts from the "lib" subdirectory, unless the
 * SELENIUM_DEV_MODE environment variable has been set to 1, in which case all
 * scripts will be loaded from the Selenium client containing this script.
 */

'use strict';

var fs = require('fs'),
    path = require('path'),
    vm = require('vm');


/**
 * If this script was loaded from the Selenium project repo, it will operate in
 * development mode, adjusting how it loads Closure-based dependencies.
 * @type {boolean}
 */
var devMode = (function() {
  var buildDescFile = path.join(__dirname, '..', 'build.desc');
  return fs.existsSync(buildDescFile);
})();


/** @return {boolean} Whether this script was loaded in dev mode. */
function isDevMode() {
  return devMode;
}


/**
 * @type {string} Path to Closure's base file, relative to this module.
 * @const
 */
var CLOSURE_BASE_FILE_PATH = (function() {
  var relativePath = isDevMode() ?
      '../../../third_party/closure/goog/base.js' :
      './lib/goog/base.js';
  return path.join(__dirname, relativePath);
})();


/**
 * @type {string} Path to Closure's base file, relative to this module.
 * @const
 */
var DEPS_FILE_PATH = (function() {
  var relativePath = isDevMode() ?
      '../../../javascript/deps.js' :
      './lib/goog/deps.js';
  return path.join(__dirname, relativePath);
})();


/**
 * Maintains a unique context for Closure library-based code.
 * @param {boolean=} opt_configureForTesting Whether to configure a fake DOM
 *     for Closure-testing code that (incorrectly) assumes a DOM is always
 *     present.
 * @constructor
 */
function Context(opt_configureForTesting) {
  var closure = this.closure = vm.createContext({
    console: console,
    setTimeout: setTimeout,
    setInterval: setInterval,
    clearTimeout: clearTimeout,
    clearInterval: clearInterval,
    process: process,
    require: require,
    Buffer: Buffer,
    Error: Error,
    CLOSURE_BASE_PATH: path.dirname(CLOSURE_BASE_FILE_PATH) + '/',
    CLOSURE_IMPORT_SCRIPT: function(src) {
      loadScript(src);
      return true;
    },
    CLOSURE_NO_DEPS: !isDevMode(),
    goog: {}
  });
  closure.window = closure.top = closure;

  if (opt_configureForTesting) {
    closure.document = {
      body: {},
      createElement: function() { return {}; },
      getElementsByTagName: function() { return []; }
    };
    closure.document.body.ownerDocument = closure.document;
  }

  loadScript(CLOSURE_BASE_FILE_PATH);
  loadScript(DEPS_FILE_PATH);

  /**
   * Synchronously loads a script into the protected Closure context.
   * @param {string} src Path to the file to load.
   */
  function loadScript(src) {
    src = path.normalize(src);
    var contents = fs.readFileSync(src, 'utf8');
    vm.runInContext(contents, closure, src);
  }
}


var context = new Context();


/**
 * Loads a symbol by name from the protected Closure context.
 * @param {string} symbol The symbol to load.
 * @return {?} The loaded symbol, or {@code null} if not found.
 * @throws {Error} If the symbol has not been defined.
 */
function closureRequire(symbol) {
  context.closure.goog.require(symbol);
  return context.closure.goog.getObjectByName(symbol);
}


// PUBLIC API


/**
 * Loads a symbol by name from the protected Closure context and exports its
 * public API to the provided object. This function relies on Closure code
 * conventions to define the public API of an object as those properties whose
 * name does not end with "_".
 * @param {string} symbol The symbol to load. This must resolve to an object.
 * @return {!Object} An object with the exported API.
 * @throws {Error} If the symbol has not been defined or does not resolve to
 *     an object.
 */
exports.exportPublicApi = function(symbol) {
  var src = closureRequire(symbol);
  if (typeof src != 'object' || src === null) {
    throw Error('"' + symbol + '" must resolve to an object');
  }

  var dest = {};
  Object.keys(src).forEach(function(key) {
    if (key[key.length - 1] != '_') {
      dest[key] = src[key];
    }
  });

  return dest;
};


if (isDevMode()) {
  exports.closure = context.closure;
}
exports.Context = Context;
exports.isDevMode = isDevMode;
exports.require = closureRequire;
