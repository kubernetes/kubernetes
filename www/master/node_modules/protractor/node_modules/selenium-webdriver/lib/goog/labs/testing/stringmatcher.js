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
 * @fileoverview Provides the built-in string matchers like containsString,
 *     startsWith, endsWith, etc.
 */


goog.provide('goog.labs.testing.ContainsStringMatcher');
goog.provide('goog.labs.testing.EndsWithMatcher');
goog.provide('goog.labs.testing.EqualToIgnoringCaseMatcher');
goog.provide('goog.labs.testing.EqualToIgnoringWhitespaceMatcher');
goog.provide('goog.labs.testing.EqualsMatcher');
goog.provide('goog.labs.testing.RegexMatcher');
goog.provide('goog.labs.testing.StartsWithMatcher');
goog.provide('goog.labs.testing.StringContainsInOrderMatcher');


goog.require('goog.asserts');
goog.require('goog.labs.testing.Matcher');
goog.require('goog.string');



/**
 * The ContainsString matcher.
 *
 * @param {string} value The expected string.
 *
 * @constructor
 * @struct
 * @implements {goog.labs.testing.Matcher}
 * @final
 */
goog.labs.testing.ContainsStringMatcher = function(value) {
  /**
   * @type {string}
   * @private
   */
  this.value_ = value;
};


/**
 * Determines if input string contains the expected string.
 *
 * @override
 */
goog.labs.testing.ContainsStringMatcher.prototype.matches =
    function(actualValue) {
  goog.asserts.assertString(actualValue);
  return goog.string.contains(actualValue, this.value_);
};


/**
 * @override
 */
goog.labs.testing.ContainsStringMatcher.prototype.describe =
    function(actualValue) {
  return actualValue + ' does not contain ' + this.value_;
};



/**
 * The EndsWith matcher.
 *
 * @param {string} value The expected string.
 *
 * @constructor
 * @struct
 * @implements {goog.labs.testing.Matcher}
 * @final
 */
goog.labs.testing.EndsWithMatcher = function(value) {
  /**
   * @type {string}
   * @private
   */
  this.value_ = value;
};


/**
 * Determines if input string ends with the expected string.
 *
 * @override
 */
goog.labs.testing.EndsWithMatcher.prototype.matches = function(actualValue) {
  goog.asserts.assertString(actualValue);
  return goog.string.endsWith(actualValue, this.value_);
};


/**
 * @override
 */
goog.labs.testing.EndsWithMatcher.prototype.describe =
    function(actualValue) {
  return actualValue + ' does not end with ' + this.value_;
};



/**
 * The EqualToIgnoringWhitespace matcher.
 *
 * @param {string} value The expected string.
 *
 * @constructor
 * @struct
 * @implements {goog.labs.testing.Matcher}
 * @final
 */
goog.labs.testing.EqualToIgnoringWhitespaceMatcher = function(value) {
  /**
   * @type {string}
   * @private
   */
  this.value_ = value;
};


/**
 * Determines if input string contains the expected string.
 *
 * @override
 */
goog.labs.testing.EqualToIgnoringWhitespaceMatcher.prototype.matches =
    function(actualValue) {
  goog.asserts.assertString(actualValue);
  var string1 = goog.string.collapseWhitespace(actualValue);

  return goog.string.caseInsensitiveCompare(this.value_, string1) === 0;
};


/**
 * @override
 */
goog.labs.testing.EqualToIgnoringWhitespaceMatcher.prototype.describe =
    function(actualValue) {
  return actualValue + ' is not equal(ignoring whitespace) to ' + this.value_;
};



/**
 * The Equals matcher.
 *
 * @param {string} value The expected string.
 *
 * @constructor
 * @struct
 * @implements {goog.labs.testing.Matcher}
 * @final
 */
goog.labs.testing.EqualsMatcher = function(value) {
  /**
   * @type {string}
   * @private
   */
  this.value_ = value;
};


/**
 * Determines if input string is equal to the expected string.
 *
 * @override
 */
goog.labs.testing.EqualsMatcher.prototype.matches = function(actualValue) {
  goog.asserts.assertString(actualValue);
  return this.value_ === actualValue;
};


/**
 * @override
 */
goog.labs.testing.EqualsMatcher.prototype.describe =
    function(actualValue) {
  return actualValue + ' is not equal to ' + this.value_;
};



/**
 * The MatchesRegex matcher.
 *
 * @param {!RegExp} regex The expected regex.
 *
 * @constructor
 * @struct
 * @implements {goog.labs.testing.Matcher}
 * @final
 */
goog.labs.testing.RegexMatcher = function(regex) {
  /**
   * @type {!RegExp}
   * @private
   */
  this.regex_ = regex;
};


/**
 * Determines if input string is equal to the expected string.
 *
 * @override
 */
goog.labs.testing.RegexMatcher.prototype.matches = function(
    actualValue) {
  goog.asserts.assertString(actualValue);
  return this.regex_.test(actualValue);
};


/**
 * @override
 */
goog.labs.testing.RegexMatcher.prototype.describe =
    function(actualValue) {
  return actualValue + ' does not match ' + this.regex_;
};



/**
 * The StartsWith matcher.
 *
 * @param {string} value The expected string.
 *
 * @constructor
 * @struct
 * @implements {goog.labs.testing.Matcher}
 * @final
 */
goog.labs.testing.StartsWithMatcher = function(value) {
  /**
   * @type {string}
   * @private
   */
  this.value_ = value;
};


/**
 * Determines if input string starts with the expected string.
 *
 * @override
 */
goog.labs.testing.StartsWithMatcher.prototype.matches = function(actualValue) {
  goog.asserts.assertString(actualValue);
  return goog.string.startsWith(actualValue, this.value_);
};


/**
 * @override
 */
goog.labs.testing.StartsWithMatcher.prototype.describe =
    function(actualValue) {
  return actualValue + ' does not start with ' + this.value_;
};



/**
 * The StringContainsInOrdermatcher.
 *
 * @param {Array.<string>} values The expected string values.
 *
 * @constructor
 * @struct
 * @implements {goog.labs.testing.Matcher}
 * @final
 */
goog.labs.testing.StringContainsInOrderMatcher = function(values) {
  /**
   * @type {Array.<string>}
   * @private
   */
  this.values_ = values;
};


/**
 * Determines if input string contains, in order, the expected array of strings.
 *
 * @override
 */
goog.labs.testing.StringContainsInOrderMatcher.prototype.matches =
    function(actualValue) {
  goog.asserts.assertString(actualValue);
  var currentIndex, previousIndex = 0;
  for (var i = 0; i < this.values_.length; i++) {
    currentIndex = goog.string.contains(actualValue, this.values_[i]);
    if (currentIndex < 0 || currentIndex < previousIndex) {
      return false;
    }
    previousIndex = currentIndex;
  }
  return true;
};


/**
 * @override
 */
goog.labs.testing.StringContainsInOrderMatcher.prototype.describe =
    function(actualValue) {
  return actualValue + ' does not contain the expected values in order.';
};


/**
 * Matches a string containing the given string.
 *
 * @param {string} value The expected value.
 *
 * @return {!goog.labs.testing.ContainsStringMatcher} A
 *     ContainsStringMatcher.
 */
function containsString(value) {
  return new goog.labs.testing.ContainsStringMatcher(value);
}


/**
 * Matches a string that ends with the given string.
 *
 * @param {string} value The expected value.
 *
 * @return {!goog.labs.testing.EndsWithMatcher} A
 *     EndsWithMatcher.
 */
function endsWith(value) {
  return new goog.labs.testing.EndsWithMatcher(value);
}


/**
 * Matches a string that equals (ignoring whitespace) the given string.
 *
 * @param {string} value The expected value.
 *
 * @return {!goog.labs.testing.EqualToIgnoringWhitespaceMatcher} A
 *     EqualToIgnoringWhitespaceMatcher.
 */
function equalToIgnoringWhitespace(value) {
  return new goog.labs.testing.EqualToIgnoringWhitespaceMatcher(value);
}


/**
 * Matches a string that equals the given string.
 *
 * @param {string} value The expected value.
 *
 * @return {!goog.labs.testing.EqualsMatcher} A EqualsMatcher.
 */
function equals(value) {
  return new goog.labs.testing.EqualsMatcher(value);
}


/**
 * Matches a string against a regular expression.
 *
 * @param {!RegExp} regex The expected regex.
 *
 * @return {!goog.labs.testing.RegexMatcher} A RegexMatcher.
 */
function matchesRegex(regex) {
  return new goog.labs.testing.RegexMatcher(regex);
}


/**
 * Matches a string that starts with the given string.
 *
 * @param {string} value The expected value.
 *
 * @return {!goog.labs.testing.StartsWithMatcher} A
 *     StartsWithMatcher.
 */
function startsWith(value) {
  return new goog.labs.testing.StartsWithMatcher(value);
}


/**
 * Matches a string that contains the given strings in order.
 *
 * @param {Array.<string>} values The expected value.
 *
 * @return {!goog.labs.testing.StringContainsInOrderMatcher} A
 *     StringContainsInOrderMatcher.
 */
function stringContainsInOrder(values) {
  return new goog.labs.testing.StringContainsInOrderMatcher(values);
}
