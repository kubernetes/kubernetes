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
 * @fileoverview Assertions and expectation utilities for use in WebDriver test
 * cases.
 */

goog.provide('webdriver.testing.Assertion');
goog.provide('webdriver.testing.ContainsMatcher');
goog.provide('webdriver.testing.NegatedAssertion');
goog.provide('webdriver.testing.assert');
goog.provide('webdriver.testing.asserts');

goog.require('goog.array');
goog.require('goog.labs.testing.CloseToMatcher');
goog.require('goog.labs.testing.EndsWithMatcher');
goog.require('goog.labs.testing.EqualToMatcher');
goog.require('goog.labs.testing.EqualsMatcher');
goog.require('goog.labs.testing.GreaterThanEqualToMatcher');
goog.require('goog.labs.testing.GreaterThanMatcher');
goog.require('goog.labs.testing.LessThanEqualToMatcher');
goog.require('goog.labs.testing.LessThanMatcher');
goog.require('goog.labs.testing.InstanceOfMatcher');
goog.require('goog.labs.testing.IsNotMatcher');
goog.require('goog.labs.testing.IsNullMatcher');
goog.require('goog.labs.testing.IsNullOrUndefinedMatcher');
goog.require('goog.labs.testing.IsUndefinedMatcher');
goog.require('goog.labs.testing.Matcher');
goog.require('goog.labs.testing.ObjectEqualsMatcher');
goog.require('goog.labs.testing.RegexMatcher');
goog.require('goog.labs.testing.StartsWithMatcher');
goog.require('goog.labs.testing.assertThat');
goog.require('goog.string');
goog.require('webdriver.promise');


/**
 * Accepts strins or array-like structures that contain {@code value}.
 * @param {*} value The value to check for.
 * @constructor
 * @implements {goog.labs.testing.Matcher}
 */
webdriver.testing.ContainsMatcher = function(value) {
  /** @private {*} */
  this.value_ = value;
};


/** @override */
webdriver.testing.ContainsMatcher.prototype.matches = function(actualValue) {
  if (goog.isString(actualValue)) {
    return goog.string.contains(
        actualValue, /** @type {string} */(this.value_));
  } else {
    return goog.array.contains(
        /** @type {goog.array.ArrayLike} */(actualValue), this.value_);
  }
};


/** @override */
webdriver.testing.ContainsMatcher.prototype.describe = function(actualValue) {
  return actualValue + ' does not contain ' + this.value_;
};



/**
 * Utility for performing assertions against a given {@code value}. If the
 * value is a {@link webdriver.promise.Promise}, this assertion will wait
 * for it to resolve before applying any matchers.
 * @param {*} value The value to wrap and apply matchers to.
 * @constructor
 */
webdriver.testing.Assertion = function(value) {

  /** @private {*} */
  this.value_ = value;

  if (!(this instanceof webdriver.testing.NegatedAssertion)) {
    /**
     * A self reference provided for writing fluent assertions:
     *     webdriver.testing.assert(x).is.equalTo(y);
     * @type {!webdriver.testing.Assertion}
     */
    this.is = this;

    /**
     * Negates any matchers applied to this instance's value:
     *     webdriver.testing.assert(x).not.equalTo(y);
     * @type {!webdriver.testing.NegatedAssertion}
     */
    this.not = new webdriver.testing.NegatedAssertion(value);
  }
};


/**
 * Wraps an object literal implementing the Matcher interface. This is used
 * to appease the Closure compiler, which will not treat an object literal as
 * implementing an interface.
 * @param {{matches: function(*): boolean, describe: function(): string}} obj
 *     The object literal to delegate to.
 * @constructor
 * @implements {goog.labs.testing.Matcher}
 * @private
 */
webdriver.testing.Assertion.DelegatingMatcher_ = function(obj) {

  /** @override */
  this.matches = function(value) {
    return obj.matches(value);
  };

  /** @override */
  this.describe = function() {
    return obj.describe();
  };
};


/**
 * Asserts that the given {@code matcher} accepts the value wrapped by this
 * instance. If the wrapped value is a promise, this function will defer
 * applying the assertion until the value has been resolved. Otherwise, it
 * will be applied immediately.
 * @param {!goog.labs.testing.Matcher} matcher The matcher to apply
 * @param {string=} opt_message A message to include if the matcher does not
 *     accept the value wrapped by this assertion.
 * @return {webdriver.promise.Promise} The deferred assertion result, or
 *     {@code null} if the assertion was immediately applied.
 * @protected
 */
webdriver.testing.Assertion.prototype.apply = function(matcher, opt_message) {
  var result = null;
  if (webdriver.promise.isPromise(this.value_)) {
    result = webdriver.promise.when(this.value_, function(value) {
      goog.labs.testing.assertThat(value, matcher, opt_message);
    });
  } else {
    goog.labs.testing.assertThat(this.value_, matcher, opt_message);
  }
  return result;
};


/**
 * Asserts that the value managed by this assertion is a number strictly
 * greater than {@code value}.
 * @param {number} value The minimum value.
 * @param {string=} opt_message A message to include if the matcher does not
 *     accept the value wrapped by this assertion.
 * @return {webdriver.promise.Promise} The assertion result.
 */
webdriver.testing.Assertion.prototype.greaterThan = function(
    value, opt_message) {
  return this.apply(
      new goog.labs.testing.GreaterThanMatcher(value), opt_message);
};


/**
 * Asserts that the value managed by this assertion is a number >= the given
 * value.
 * @param {number} value The minimum value.
 * @param {string=} opt_message A message to include if the matcher does not
 *     accept the value wrapped by this assertion.
 * @return {webdriver.promise.Promise} The assertion result.
 */
webdriver.testing.Assertion.prototype.greaterThanEqualTo = function(
    value, opt_message) {
  return this.apply(
      new goog.labs.testing.GreaterThanEqualToMatcher(value), opt_message);
};


/**
 * Asserts that the value managed by this assertion is a number strictly less
 * than the given value.
 * @param {number} value The maximum value.
 * @param {string=} opt_message A message to include if the matcher does not
 *     accept the value wrapped by this assertion.
 * @return {webdriver.promise.Promise} The assertion result.
 */
webdriver.testing.Assertion.prototype.lessThan = function(value, opt_message) {
  return this.apply(
      new goog.labs.testing.LessThanMatcher(value), opt_message);
};


/**
 * Asserts that the value managed by this assertion is a number <= the given
 * value.
 * @param {number} value The maximum value.
 * @param {string=} opt_message A message to include if the matcher does not
 *     accept the value wrapped by this assertion.
 * @return {webdriver.promise.Promise} The assertion result.
 */
webdriver.testing.Assertion.prototype.lessThanEqualTo = function(
    value, opt_message) {
  return this.apply(
      new goog.labs.testing.LessThanEqualToMatcher(value), opt_message);
};


/**
 * Asserts that the wrapped value is a number within a given distance of an
 * expected value.
 * @param {number} value The expected value.
 * @param {number} range The maximum amount the actual value is permitted to
 *     differ from the expected value.
 * @param {string=} opt_message A message to include if the matcher does not
 *     accept the value wrapped by this assertion.
 * @return {webdriver.promise.Promise} The assertion result.
 */
webdriver.testing.Assertion.prototype.closeTo = function(
    value, range, opt_message) {
  return this.apply(
      new goog.labs.testing.CloseToMatcher(value, range), opt_message);
};


/**
 * Asserts that the wrapped value is an instance of the given class.
 * @param {!Function} ctor The expected class constructor.
 * @param {string=} opt_message A message to include if the matcher does not
 *     accept the value wrapped by this assertion.
 * @return {webdriver.promise.Promise} The assertion result.
 */
webdriver.testing.Assertion.prototype.instanceOf = function(ctor, opt_message) {
  return this.apply(
      new goog.labs.testing.InstanceOfMatcher(ctor), opt_message);
};


/**
 * Asserts that the wrapped value is null.
 * @param {string=} opt_message A message to include if the matcher does not
 *     accept the value wrapped by this assertion.
 * @return {webdriver.promise.Promise} The assertion result.
 */
webdriver.testing.Assertion.prototype.isNull = function(opt_message) {
  return this.apply(new goog.labs.testing.IsNullMatcher(), opt_message);
};


/**
 * Asserts that the wrapped value is undefined.
 * @param {string=} opt_message A message to include if the matcher does not
 *     accept the value wrapped by this assertion.
 * @return {webdriver.promise.Promise} The assertion result.
 */
webdriver.testing.Assertion.prototype.isUndefined = function(opt_message) {
  return this.apply(new goog.labs.testing.IsUndefinedMatcher(), opt_message);
};


/**
 * Asserts that the wrapped value is null or undefined.
 * @param {string=} opt_message A message to include if the matcher does not
 *     accept the value wrapped by this assertion.
 * @return {webdriver.promise.Promise} The assertion result.
 */
webdriver.testing.Assertion.prototype.isNullOrUndefined = function(
    opt_message) {
  return this.apply(
      new goog.labs.testing.IsNullOrUndefinedMatcher(), opt_message);
};


/**
 * Asserts that the wrapped value is a string or array-like structure
 * containing the given value.
 * @param {*} value The expected value.
 * @param {string=} opt_message A message to include if the matcher does not
 *     accept the value wrapped by this assertion.
 * @return {webdriver.promise.Promise} The assertion result.
 */
webdriver.testing.Assertion.prototype.contains = function(value, opt_message) {
  return this.apply(
      new webdriver.testing.ContainsMatcher(value), opt_message);
};


/**
 * Asserts that the wrapped value is a string ending with the given suffix.
 * @param {string} suffix The expected suffix.
 * @param {string=} opt_message A message to include if the matcher does not
 *     accept the value wrapped by this assertion.
 * @return {webdriver.promise.Promise} The assertion result.
 */
webdriver.testing.Assertion.prototype.endsWith = function(
    suffix, opt_message) {
  return this.apply(
      new goog.labs.testing.EndsWithMatcher(suffix), opt_message);
};


/**
 * Asserts that the wrapped value is a string starting with the given prefix.
 * @param {string} prefix The expected prefix.
 * @param {string=} opt_message A message to include if the matcher does not
 *     accept the value wrapped by this assertion.
 * @return {webdriver.promise.Promise} The assertion result.
 */
webdriver.testing.Assertion.prototype.startsWith = function(
    prefix, opt_message) {
  return this.apply(
      new goog.labs.testing.StartsWithMatcher(prefix), opt_message);
};


/**
 * Asserts that the wrapped value is a string that matches the given RegExp.
 * @param {!RegExp} regex The regex to test.
 * @param {string=} opt_message A message to include if the matcher does not
 *     accept the value wrapped by this assertion.
 * @return {webdriver.promise.Promise} The assertion result.
 */
webdriver.testing.Assertion.prototype.matches = function(regex, opt_message) {
  return this.apply(new goog.labs.testing.RegexMatcher(regex), opt_message);
};


/**
 * Asserts that the value managed by this assertion is strictly equal to the
 * given {@code value}.
 * @param {*} value The expected value.
 * @param {string=} opt_message A message to include if the matcher does not
 *     accept the value wrapped by this assertion.
 * @return {webdriver.promise.Promise} The assertion result.
 */
webdriver.testing.Assertion.prototype.equalTo = function(value, opt_message) {
  return this.apply(webdriver.testing.asserts.equalTo(value), opt_message);
};


/**
 * Asserts that the value managed by this assertion is strictly true.
 * @return {webdriver.promise.Promise} The assertion result.
 */
webdriver.testing.Assertion.prototype.isTrue = function() {
  return this.equalTo(true);
};


/**
 * Asserts that the value managed by this assertion is strictly false.
 * @return {webdriver.promise.Promise} The assertion result.
 */
webdriver.testing.Assertion.prototype.isFalse = function() {
  return this.equalTo(false);
};



/**
 * An assertion that negates any applied matchers.
 * @param {*} value The value to perform assertions on.
 * @constructor
 * @extends {webdriver.testing.Assertion}
 */
webdriver.testing.NegatedAssertion = function(value) {
  goog.base(this, value);
  this.value = value;
};
goog.inherits(
    webdriver.testing.NegatedAssertion, webdriver.testing.Assertion);


/** @override */
webdriver.testing.NegatedAssertion.prototype.apply = function(
    matcher, opt_message) {
  matcher = new goog.labs.testing.IsNotMatcher(matcher);
  return goog.base(this, 'apply', matcher, opt_message);
};



/**
 * Creates a new assertion.
 * @param {*} value The value to perform an assertion on.
 * @return {!webdriver.testing.Assertion} The new assertion.
 */
webdriver.testing.assert = function(value) {
  return new webdriver.testing.Assertion(value);
};


/**
 * Registers a new assertion to expose from the
 * {@link webdriver.testing.Assertion} prototype.
 * @param {string} name The assertion name.
 * @param {(function(new: goog.labs.testing.Matcher, *)|
 *          {matches: function(*): boolean,
 *           describe: function(): string})} matcherTemplate Either the
 *     matcher constructor to use, or an object literal defining a matcher.
 */
webdriver.testing.assert.register = function(name, matcherTemplate) {
  webdriver.testing.Assertion.prototype[name] = function(value, opt_message) {
    var matcher;
    if (goog.isFunction(matcherTemplate)) {
      var ctor = /** @type {function(new: goog.labs.testing.Matcher, *)} */ (
          matcherTemplate);
      matcher = new ctor(value);
    } else {
      matcher = new webdriver.testing.Assertion.DelegatingMatcher_(value);
    }
    return this.apply(matcher, opt_message);
  };
};


/**
 * Asserts that a matcher accepts a given value.  This function has two
 * signatures based on the number of arguments:
 *
 * Two arguments:
 *   assertThat(actualValue, matcher)
 * Three arguments:
 *   assertThat(failureMessage, actualValue, matcher)
 *
 * @param {*} failureMessageOrActualValue Either a failure message or the value
 *     to apply to the given matcher.
 * @param {*} actualValueOrMatcher Either the value to apply to the given
 *     matcher, or the matcher itself.
 * @param {goog.labs.testing.Matcher=} opt_matcher The matcher to use;
 *     ignored unless this function is invoked with three arguments.
 * @return {!webdriver.promise.Promise} The assertion result.
 * @deprecated Use webdriver.testing.asserts.assert instead.
 */
webdriver.testing.asserts.assertThat = function(
    failureMessageOrActualValue, actualValueOrMatcher, opt_matcher) {
  var args = goog.array.slice(arguments, 0);

  var message = args.length > 2 ? args.shift() : '';
  if (message) message += '\n';

  var actualValue = args.shift();
  var matcher = args.shift();

  return webdriver.promise.when(actualValue, function(value) {
    goog.labs.testing.assertThat(value, matcher, message);
  });
};


/**
 * Creates an equality matcher.
 * @param {*} expected The expected value.
 * @return {!goog.labs.testing.Matcher} The new matcher.
 */
webdriver.testing.asserts.equalTo = function(expected) {
  if (goog.isString(expected)) {
    return new goog.labs.testing.EqualsMatcher(expected);
  } else if (goog.isNumber(expected)) {
    return new goog.labs.testing.EqualToMatcher(expected);
  } else {
    return new goog.labs.testing.ObjectEqualsMatcher(
        /** @type {!Object} */ (expected));
  }
};


goog.exportSymbol('assertThat', webdriver.testing.asserts.assertThat);
// Mappings for goog.labs.testing matcher functions to the legacy
// webdriver.testing.asserts matchers.
goog.exportSymbol('contains', containsString);
goog.exportSymbol('equalTo', webdriver.testing.asserts.equalTo);
goog.exportSymbol('equals', webdriver.testing.asserts.equalTo);
goog.exportSymbol('is', webdriver.testing.asserts.equalTo);
goog.exportSymbol('not', isNot);
goog.exportSymbol('or', anyOf);
