// Copyright 2009 The Closure Library Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS-IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Helper for passing property names as string literals in
 * compiled test code.
 *
 */

goog.provide('goog.testing.ObjectPropertyString');



/**
 * Object to pass a property name as a string literal and its containing object
 * when the JSCompiler is rewriting these names. This should only be used in
 * test code.
 *
 * @param {Object} object The containing object.
 * @param {Object|string} propertyString Property name as a string literal.
 * @constructor
 * @final
 */
goog.testing.ObjectPropertyString = function(object, propertyString) {
  this.object_ = object;
  this.propertyString_ = /** @type {string} */ (propertyString);
};


/**
 * @type {Object}
 * @private
 */
goog.testing.ObjectPropertyString.prototype.object_;


/**
 * @type {string}
 * @private
 */
goog.testing.ObjectPropertyString.prototype.propertyString_;


/**
 * @return {Object} The object.
 */
goog.testing.ObjectPropertyString.prototype.getObject = function() {
  return this.object_;
};


/**
 * @return {string} The property string.
 */
goog.testing.ObjectPropertyString.prototype.getPropertyString = function() {
  return this.propertyString_;
};
