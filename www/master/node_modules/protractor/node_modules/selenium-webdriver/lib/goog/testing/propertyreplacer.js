// Copyright 2008 The Closure Library Authors. All Rights Reserved.
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
 * @fileoverview Helper class for creating stubs for testing.
 *
 */

goog.provide('goog.testing.PropertyReplacer');

/** @suppress {extraRequire} Needed for some tests to compile. */
goog.require('goog.testing.ObjectPropertyString');
goog.require('goog.userAgent');



/**
 * Helper class for stubbing out variables and object properties for unit tests.
 * This class can change the value of some variables before running the test
 * cases, and to reset them in the tearDown phase.
 * See googletest.StubOutForTesting as an analogy in Python:
 * http://protobuf.googlecode.com/svn/trunk/python/stubout.py
 *
 * Example usage:
 * <pre>var stubs = new goog.testing.PropertyReplacer();
 *
 * function setUp() {
 *   // Mock functions used in all test cases.
 *   stubs.set(Math, 'random', function() {
 *     return 4;  // Chosen by fair dice roll. Guaranteed to be random.
 *   });
 * }
 *
 * function tearDown() {
 *   stubs.reset();
 * }
 *
 * function testThreeDice() {
 *   // Mock a constant used only in this test case.
 *   stubs.set(goog.global, 'DICE_COUNT', 3);
 *   assertEquals(12, rollAllDice());
 * }</pre>
 *
 * Constraints on altered objects:
 * <ul>
 *   <li>DOM subclasses aren't supported.
 *   <li>The value of the objects' constructor property must either be equal to
 *       the real constructor or kept untouched.
 * </ul>
 *
 * @constructor
 * @final
 */
goog.testing.PropertyReplacer = function() {
  /**
   * Stores the values changed by the set() method in chronological order.
   * Its items are objects with 3 fields: 'object', 'key', 'value'. The
   * original value for the given key in the given object is stored under the
   * 'value' key.
   * @type {Array.<Object>}
   * @private
   */
  this.original_ = [];
};


/**
 * Indicates that a key didn't exist before having been set by the set() method.
 * @type {Object}
 * @private
 */
goog.testing.PropertyReplacer.NO_SUCH_KEY_ = {};


/**
 * Tells if the given key exists in the object. Ignores inherited fields.
 * @param {Object|Function} obj The JavaScript or native object or function
 *     whose key is to be checked.
 * @param {string} key The key to check.
 * @return {boolean} Whether the object has the key as own key.
 * @private
 */
goog.testing.PropertyReplacer.hasKey_ = function(obj, key) {
  if (!(key in obj)) {
    return false;
  }
  // hasOwnProperty is only reliable with JavaScript objects. It returns false
  // for built-in DOM attributes.
  if (Object.prototype.hasOwnProperty.call(obj, key)) {
    return true;
  }
  // In all browsers except Opera obj.constructor never equals to Object if
  // obj is an instance of a native class. In Opera we have to fall back on
  // examining obj.toString().
  if (obj.constructor == Object &&
      (!goog.userAgent.OPERA ||
          Object.prototype.toString.call(obj) == '[object Object]')) {
    return false;
  }
  try {
    // Firefox hack to consider "className" part of the HTML elements or
    // "body" part of document. Although they are defined in the prototype of
    // HTMLElement or Document, accessing them this way throws an exception.
    // <pre>
    //   var dummy = document.body.constructor.prototype.className
    //   [Exception... "Cannot modify properties of a WrappedNative"]
    // </pre>
    var dummy = obj.constructor.prototype[key];
  } catch (e) {
    return true;
  }
  return !(key in obj.constructor.prototype);
};


/**
 * Deletes a key from an object. Sets it to undefined or empty string if the
 * delete failed.
 * @param {Object|Function} obj The object or function to delete a key from.
 * @param {string} key The key to delete.
 * @private
 */
goog.testing.PropertyReplacer.deleteKey_ = function(obj, key) {
  try {
    delete obj[key];
    // Delete has no effect for built-in properties of DOM nodes in FF.
    if (!goog.testing.PropertyReplacer.hasKey_(obj, key)) {
      return;
    }
  } catch (e) {
    // IE throws TypeError when trying to delete properties of native objects
    // (e.g. DOM nodes or window), even if they have been added by JavaScript.
  }

  obj[key] = undefined;
  if (obj[key] == 'undefined') {
    // Some properties such as className in IE are always evaluated as string
    // so undefined will become 'undefined'.
    obj[key] = '';
  }
};


/**
 * Adds or changes a value in an object while saving its original state.
 * @param {Object|Function} obj The JavaScript or native object or function to
 *     alter. See the constraints in the class description.
 * @param {string} key The key to change the value for.
 * @param {*} value The new value to set.
 */
goog.testing.PropertyReplacer.prototype.set = function(obj, key, value) {
  var origValue = goog.testing.PropertyReplacer.hasKey_(obj, key) ? obj[key] :
                  goog.testing.PropertyReplacer.NO_SUCH_KEY_;
  this.original_.push({object: obj, key: key, value: origValue});
  obj[key] = value;
};


/**
 * Changes an existing value in an object to another one of the same type while
 * saving its original state. The advantage of {@code replace} over {@link #set}
 * is that {@code replace} protects against typos and erroneously passing tests
 * after some members have been renamed during a refactoring.
 * @param {Object|Function} obj The JavaScript or native object or function to
 *     alter. See the constraints in the class description.
 * @param {string} key The key to change the value for. It has to be present
 *     either in {@code obj} or in its prototype chain.
 * @param {*} value The new value to set. It has to have the same type as the
 *     original value. The types are compared with {@link goog.typeOf}.
 * @throws {Error} In case of missing key or type mismatch.
 */
goog.testing.PropertyReplacer.prototype.replace = function(obj, key, value) {
  if (!(key in obj)) {
    throw Error('Cannot replace missing property "' + key + '" in ' + obj);
  }
  if (goog.typeOf(obj[key]) != goog.typeOf(value)) {
    throw Error('Cannot replace property "' + key + '" in ' + obj +
        ' with a value of different type');
  }
  this.set(obj, key, value);
};


/**
 * Builds an object structure for the provided namespace path.  Doesn't
 * overwrite those prefixes of the path that are already objects or functions.
 * @param {string} path The path to create or alter, e.g. 'goog.ui.Menu'.
 * @param {*} value The value to set.
 */
goog.testing.PropertyReplacer.prototype.setPath = function(path, value) {
  var parts = path.split('.');
  var obj = goog.global;
  for (var i = 0; i < parts.length - 1; i++) {
    var part = parts[i];
    if (part == 'prototype' && !obj[part]) {
      throw Error('Cannot set the prototype of ' + parts.slice(0, i).join('.'));
    }
    if (!goog.isObject(obj[part]) && !goog.isFunction(obj[part])) {
      this.set(obj, part, {});
    }
    obj = obj[part];
  }
  this.set(obj, parts[parts.length - 1], value);
};


/**
 * Deletes the key from the object while saving its original value.
 * @param {Object|Function} obj The JavaScript or native object or function to
 *     alter. See the constraints in the class description.
 * @param {string} key The key to delete.
 */
goog.testing.PropertyReplacer.prototype.remove = function(obj, key) {
  if (goog.testing.PropertyReplacer.hasKey_(obj, key)) {
    this.original_.push({object: obj, key: key, value: obj[key]});
    goog.testing.PropertyReplacer.deleteKey_(obj, key);
  }
};


/**
 * Resets all changes made by goog.testing.PropertyReplacer.prototype.set.
 */
goog.testing.PropertyReplacer.prototype.reset = function() {
  for (var i = this.original_.length - 1; i >= 0; i--) {
    var original = this.original_[i];
    if (original.value == goog.testing.PropertyReplacer.NO_SUCH_KEY_) {
      goog.testing.PropertyReplacer.deleteKey_(original.object, original.key);
    } else {
      original.object[original.key] = original.value;
    }
    delete this.original_[i];
  }
  this.original_.length = 0;
};
