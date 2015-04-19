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
 * @fileoverview Provides the built-in number matchers like lessThan,
 * greaterThan, etc.
 */


goog.provide('goog.labs.testing.CloseToMatcher');
goog.provide('goog.labs.testing.EqualToMatcher');
goog.provide('goog.labs.testing.GreaterThanEqualToMatcher');
goog.provide('goog.labs.testing.GreaterThanMatcher');
goog.provide('goog.labs.testing.LessThanEqualToMatcher');
goog.provide('goog.labs.testing.LessThanMatcher');


goog.require('goog.asserts');
goog.require('goog.labs.testing.Matcher');



/**
 * The GreaterThan matcher.
 *
 * @param {number} value The value to compare.
 *
 * @constructor
 * @struct
 * @implements {goog.labs.testing.Matcher}
 * @final
 */
goog.labs.testing.GreaterThanMatcher = function(value) {
  /**
   * @type {number}
   * @private
   */
  this.value_ = value;
};


/**
 * Determines if input value is greater than the expected value.
 *
 * @override
 */
goog.labs.testing.GreaterThanMatcher.prototype.matches = function(actualValue) {
  goog.asserts.assertNumber(actualValue);
  return actualValue > this.value_;
};


/**
 * @override
 */
goog.labs.testing.GreaterThanMatcher.prototype.describe =
    function(actualValue) {
  goog.asserts.assertNumber(actualValue);
  return actualValue + ' is not greater than ' + this.value_;
};



/**
 * The lessThan matcher.
 *
 * @param {number} value The value to compare.
 *
 * @constructor
 * @struct
 * @implements {goog.labs.testing.Matcher}
 * @final
 */
goog.labs.testing.LessThanMatcher = function(value) {
  /**
   * @type {number}
   * @private
   */
  this.value_ = value;
};


/**
 * Determines if the input value is less than the expected value.
 *
 * @override
 */
goog.labs.testing.LessThanMatcher.prototype.matches = function(actualValue) {
  goog.asserts.assertNumber(actualValue);
  return actualValue < this.value_;
};


/**
 * @override
 */
goog.labs.testing.LessThanMatcher.prototype.describe =
    function(actualValue) {
  goog.asserts.assertNumber(actualValue);
  return actualValue + ' is not less than ' + this.value_;
};



/**
 * The GreaterThanEqualTo matcher.
 *
 * @param {number} value The value to compare.
 *
 * @constructor
 * @struct
 * @implements {goog.labs.testing.Matcher}
 * @final
 */
goog.labs.testing.GreaterThanEqualToMatcher = function(value) {
  /**
   * @type {number}
   * @private
   */
  this.value_ = value;
};


/**
 * Determines if the input value is greater than equal to the expected value.
 *
 * @override
 */
goog.labs.testing.GreaterThanEqualToMatcher.prototype.matches =
    function(actualValue) {
  goog.asserts.assertNumber(actualValue);
  return actualValue >= this.value_;
};


/**
 * @override
 */
goog.labs.testing.GreaterThanEqualToMatcher.prototype.describe =
    function(actualValue) {
  goog.asserts.assertNumber(actualValue);
  return actualValue + ' is not greater than equal to ' + this.value_;
};



/**
 * The LessThanEqualTo matcher.
 *
 * @param {number} value The value to compare.
 *
 * @constructor
 * @struct
 * @implements {goog.labs.testing.Matcher}
 * @final
 */
goog.labs.testing.LessThanEqualToMatcher = function(value) {
  /**
   * @type {number}
   * @private
   */
  this.value_ = value;
};


/**
 * Determines if the input value is less than or equal to the expected value.
 *
 * @override
 */
goog.labs.testing.LessThanEqualToMatcher.prototype.matches =
    function(actualValue) {
  goog.asserts.assertNumber(actualValue);
  return actualValue <= this.value_;
};


/**
 * @override
 */
goog.labs.testing.LessThanEqualToMatcher.prototype.describe =
    function(actualValue) {
  goog.asserts.assertNumber(actualValue);
  return actualValue + ' is not less than equal to ' + this.value_;
};



/**
 * The EqualTo matcher.
 *
 * @param {number} value The value to compare.
 *
 * @constructor
 * @struct
 * @implements {goog.labs.testing.Matcher}
 * @final
 */
goog.labs.testing.EqualToMatcher = function(value) {
  /**
   * @type {number}
   * @private
   */
  this.value_ = value;
};


/**
 * Determines if the input value is equal to the expected value.
 *
 * @override
 */
goog.labs.testing.EqualToMatcher.prototype.matches = function(actualValue) {
  goog.asserts.assertNumber(actualValue);
  return actualValue === this.value_;
};


/**
 * @override
 */
goog.labs.testing.EqualToMatcher.prototype.describe =
    function(actualValue) {
  goog.asserts.assertNumber(actualValue);
  return actualValue + ' is not equal to ' + this.value_;
};



/**
 * The CloseTo matcher.
 *
 * @param {number} value The value to compare.
 * @param {number} range The range to check within.
 *
 * @constructor
 * @struct
 * @implements {goog.labs.testing.Matcher}
 * @final
 */
goog.labs.testing.CloseToMatcher = function(value, range) {
  /**
   * @type {number}
   * @private
   */
  this.value_ = value;
  /**
   * @type {number}
   * @private
   */
  this.range_ = range;
};


/**
 * Determines if input value is within a certain range of the expected value.
 *
 * @override
 */
goog.labs.testing.CloseToMatcher.prototype.matches = function(actualValue) {
  goog.asserts.assertNumber(actualValue);
  return Math.abs(this.value_ - actualValue) < this.range_;
};


/**
 * @override
 */
goog.labs.testing.CloseToMatcher.prototype.describe =
    function(actualValue) {
  goog.asserts.assertNumber(actualValue);
  return actualValue + ' is not close to(' + this.range_ + ') ' + this.value_;
};


/**
 * @param {number} value The expected value.
 *
 * @return {!goog.labs.testing.GreaterThanMatcher} A GreaterThanMatcher.
 */
function greaterThan(value) {
  return new goog.labs.testing.GreaterThanMatcher(value);
}


/**
 * @param {number} value The expected value.
 *
 * @return {!goog.labs.testing.GreaterThanEqualToMatcher} A
 *     GreaterThanEqualToMatcher.
 */
function greaterThanEqualTo(value) {
  return new goog.labs.testing.GreaterThanEqualToMatcher(value);
}


/**
 * @param {number} value The expected value.
 *
 * @return {!goog.labs.testing.LessThanMatcher} A LessThanMatcher.
 */
function lessThan(value) {
  return new goog.labs.testing.LessThanMatcher(value);
}


/**
 * @param {number} value The expected value.
 *
 * @return {!goog.labs.testing.LessThanEqualToMatcher} A LessThanEqualToMatcher.
 */
function lessThanEqualTo(value) {
  return new goog.labs.testing.LessThanEqualToMatcher(value);
}


/**
 * @param {number} value The expected value.
 *
 * @return {!goog.labs.testing.EqualToMatcher} An EqualToMatcher.
 */
function equalTo(value) {
  return new goog.labs.testing.EqualToMatcher(value);
}


/**
 * @param {number} value The expected value.
 * @param {number} range The maximum allowed difference from the expected value.
 *
 * @return {!goog.labs.testing.CloseToMatcher} A CloseToMatcher.
 */
function closeTo(value, range) {
  return new goog.labs.testing.CloseToMatcher(value, range);
}
