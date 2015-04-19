// Copyright 2012 The Closure Library Authors. All Rights Reserved.
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
 * @fileoverview Provides the built-in logic matchers: anyOf, allOf, and isNot.
 *
 */


goog.provide('goog.labs.testing.AllOfMatcher');
goog.provide('goog.labs.testing.AnyOfMatcher');
goog.provide('goog.labs.testing.IsNotMatcher');


goog.require('goog.array');
goog.require('goog.labs.testing.Matcher');



/**
 * The AllOf matcher.
 *
 * @param {!Array.<!goog.labs.testing.Matcher>} matchers Input matchers.
 *
 * @constructor
 * @struct
 * @implements {goog.labs.testing.Matcher}
 * @final
 */
goog.labs.testing.AllOfMatcher = function(matchers) {
  /**
   * @type {!Array.<!goog.labs.testing.Matcher>}
   * @private
   */
  this.matchers_ = matchers;
};


/**
 * Determines if all of the matchers match the input value.
 *
 * @override
 */
goog.labs.testing.AllOfMatcher.prototype.matches = function(actualValue) {
  return goog.array.every(this.matchers_, function(matcher) {
    return matcher.matches(actualValue);
  });
};


/**
 * Describes why the matcher failed. The returned string is a concatenation of
 * all the failed matchers' error strings.
 *
 * @override
 */
goog.labs.testing.AllOfMatcher.prototype.describe =
    function(actualValue) {
  // TODO(user) : Optimize this to remove duplication with matches ?
  var errorString = '';
  goog.array.forEach(this.matchers_, function(matcher) {
    if (!matcher.matches(actualValue)) {
      errorString += matcher.describe(actualValue) + '\n';
    }
  });
  return errorString;
};



/**
 * The AnyOf matcher.
 *
 * @param {!Array.<!goog.labs.testing.Matcher>} matchers Input matchers.
 *
 * @constructor
 * @struct
 * @implements {goog.labs.testing.Matcher}
 * @final
 */
goog.labs.testing.AnyOfMatcher = function(matchers) {
  /**
   * @type {!Array.<!goog.labs.testing.Matcher>}
   * @private
   */
  this.matchers_ = matchers;
};


/**
 * Determines if any of the matchers matches the input value.
 *
 * @override
 */
goog.labs.testing.AnyOfMatcher.prototype.matches = function(actualValue) {
  return goog.array.some(this.matchers_, function(matcher) {
    return matcher.matches(actualValue);
  });
};


/**
 * Describes why the matcher failed.
 *
 * @override
 */
goog.labs.testing.AnyOfMatcher.prototype.describe =
    function(actualValue) {
  // TODO(user) : Optimize this to remove duplication with matches ?
  var errorString = '';
  goog.array.forEach(this.matchers_, function(matcher) {
    if (!matcher.matches(actualValue)) {
      errorString += matcher.describe(actualValue) + '\n';
    }
  });
  return errorString;
};



/**
 * The IsNot matcher.
 *
 * @param {!goog.labs.testing.Matcher} matcher The matcher to negate.
 *
 * @constructor
 * @struct
 * @implements {goog.labs.testing.Matcher}
 * @final
 */
goog.labs.testing.IsNotMatcher = function(matcher) {
  /**
   * @type {!goog.labs.testing.Matcher}
   * @private
   */
  this.matcher_ = matcher;
};


/**
 * Determines if the input value doesn't satisfy a matcher.
 *
 * @override
 */
goog.labs.testing.IsNotMatcher.prototype.matches = function(actualValue) {
  return !this.matcher_.matches(actualValue);
};


/**
 * Describes why the matcher failed.
 *
 * @override
 */
goog.labs.testing.IsNotMatcher.prototype.describe =
    function(actualValue) {
  return 'The following is false: ' + this.matcher_.describe(actualValue);
};


/**
 * Creates a matcher that will succeed only if all of the given matchers
 * succeed.
 *
 * @param {...goog.labs.testing.Matcher} var_args The matchers to test
 *     against.
 *
 * @return {!goog.labs.testing.AllOfMatcher} The AllOf matcher.
 */
function allOf(var_args) {
  var matchers = goog.array.toArray(arguments);
  return new goog.labs.testing.AllOfMatcher(matchers);
}


/**
 * Accepts a set of matchers and returns a matcher which matches
 * values which satisfy the constraints of any of the given matchers.
 *
 * @param {...goog.labs.testing.Matcher} var_args The matchers to test
 *     against.
 *
 * @return {!goog.labs.testing.AnyOfMatcher} The AnyOf matcher.
 */
function anyOf(var_args) {
  var matchers = goog.array.toArray(arguments);
  return new goog.labs.testing.AnyOfMatcher(matchers);
}


/**
 * Returns a matcher that negates the input matcher. The returned
 * matcher matches the values not matched by the input matcher and vice-versa.
 *
 * @param {!goog.labs.testing.Matcher} matcher The matcher to test against.
 *
 * @return {!goog.labs.testing.IsNotMatcher} The IsNot matcher.
 */
function isNot(matcher) {
  return new goog.labs.testing.IsNotMatcher(matcher);
}
