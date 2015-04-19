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
 * @fileoverview Matchers to be used with the mock utilities.  They allow for
 * flexible matching by type.  Custom matchers can be created by passing a
 * matcher function into an ArgumentMatcher instance.
 *
 * For examples, please see the unit test.
 *
 */


goog.provide('goog.testing.mockmatchers');
goog.provide('goog.testing.mockmatchers.ArgumentMatcher');
goog.provide('goog.testing.mockmatchers.IgnoreArgument');
goog.provide('goog.testing.mockmatchers.InstanceOf');
goog.provide('goog.testing.mockmatchers.ObjectEquals');
goog.provide('goog.testing.mockmatchers.RegexpMatch');
goog.provide('goog.testing.mockmatchers.SaveArgument');
goog.provide('goog.testing.mockmatchers.TypeOf');

goog.require('goog.array');
goog.require('goog.dom');
goog.require('goog.testing.asserts');



/**
 * A simple interface for executing argument matching.  A match in this case is
 * testing to see if a supplied object fits a given criteria.  True is returned
 * if the given criteria is met.
 * @param {Function=} opt_matchFn A function that evaluates a given argument
 *     and returns true if it meets a given criteria.
 * @param {?string=} opt_matchName The name expressing intent as part of
 *      an error message for when a match fails.
 * @constructor
 */
goog.testing.mockmatchers.ArgumentMatcher =
    function(opt_matchFn, opt_matchName) {
  /**
   * A function that evaluates a given argument and returns true if it meets a
   * given criteria.
   * @type {Function}
   * @private
   */
  this.matchFn_ = opt_matchFn || null;

  /**
   * A string indicating the match intent (e.g. isBoolean or isString).
   * @type {?string}
   * @private
   */
  this.matchName_ = opt_matchName || null;
};


/**
 * A function that takes a match argument and an optional MockExpectation
 * which (if provided) will get error information and returns whether or
 * not it matches.
 * @param {*} toVerify The argument that should be verified.
 * @param {goog.testing.MockExpectation?=} opt_expectation The expectation
 *     for this match.
 * @return {boolean} Whether or not a given argument passes verification.
 */
goog.testing.mockmatchers.ArgumentMatcher.prototype.matches =
    function(toVerify, opt_expectation) {
  if (this.matchFn_) {
    var isamatch = this.matchFn_(toVerify);
    if (!isamatch && opt_expectation) {
      if (this.matchName_) {
        opt_expectation.addErrorMessage('Expected: ' +
            this.matchName_ + ' but was: ' + _displayStringForValue(toVerify));
      } else {
        opt_expectation.addErrorMessage('Expected: missing mockmatcher' +
            ' description but was: ' +
            _displayStringForValue(toVerify));
      }
    }
    return isamatch;
  } else {
    throw Error('No match function defined for this mock matcher');
  }
};



/**
 * A matcher that verifies that an argument is an instance of a given class.
 * @param {Function} ctor The class that will be used for verification.
 * @constructor
 * @extends {goog.testing.mockmatchers.ArgumentMatcher}
 * @final
 */
goog.testing.mockmatchers.InstanceOf = function(ctor) {
  goog.testing.mockmatchers.ArgumentMatcher.call(this,
      function(obj) {
        return obj instanceof ctor;
        // NOTE: Browser differences on ctor.toString() output
        // make using that here problematic. So for now, just let
        // people know the instanceOf() failed without providing
        // browser specific details...
      }, 'instanceOf()');
};
goog.inherits(goog.testing.mockmatchers.InstanceOf,
    goog.testing.mockmatchers.ArgumentMatcher);



/**
 * A matcher that verifies that an argument is of a given type (e.g. "object").
 * @param {string} type The type that a given argument must have.
 * @constructor
 * @extends {goog.testing.mockmatchers.ArgumentMatcher}
 * @final
 */
goog.testing.mockmatchers.TypeOf = function(type) {
  goog.testing.mockmatchers.ArgumentMatcher.call(this,
      function(obj) {
        return goog.typeOf(obj) == type;
      }, 'typeOf(' + type + ')');
};
goog.inherits(goog.testing.mockmatchers.TypeOf,
    goog.testing.mockmatchers.ArgumentMatcher);



/**
 * A matcher that verifies that an argument matches a given RegExp.
 * @param {RegExp} regexp The regular expression that the argument must match.
 * @constructor
 * @extends {goog.testing.mockmatchers.ArgumentMatcher}
 * @final
 */
goog.testing.mockmatchers.RegexpMatch = function(regexp) {
  goog.testing.mockmatchers.ArgumentMatcher.call(this,
      function(str) {
        return regexp.test(str);
      }, 'match(' + regexp + ')');
};
goog.inherits(goog.testing.mockmatchers.RegexpMatch,
    goog.testing.mockmatchers.ArgumentMatcher);



/**
 * A matcher that always returns true. It is useful when the user does not care
 * for some arguments.
 * For example: mockFunction('username', 'password', IgnoreArgument);
 * @constructor
 * @extends {goog.testing.mockmatchers.ArgumentMatcher}
 * @final
 */
goog.testing.mockmatchers.IgnoreArgument = function() {
  goog.testing.mockmatchers.ArgumentMatcher.call(this,
      function() {
        return true;
      }, 'true');
};
goog.inherits(goog.testing.mockmatchers.IgnoreArgument,
    goog.testing.mockmatchers.ArgumentMatcher);



/**
 * A matcher that verifies that the argument is an object that equals the given
 * expected object, using a deep comparison.
 * @param {Object} expectedObject An object to match against when
 *     verifying the argument.
 * @constructor
 * @extends {goog.testing.mockmatchers.ArgumentMatcher}
 */
goog.testing.mockmatchers.ObjectEquals = function(expectedObject) {
  goog.testing.mockmatchers.ArgumentMatcher.call(this,
      function(matchObject) {
        assertObjectEquals('Expected equal objects', expectedObject,
            matchObject);
        return true;
      }, 'objectEquals(' + expectedObject + ')');
};
goog.inherits(goog.testing.mockmatchers.ObjectEquals,
    goog.testing.mockmatchers.ArgumentMatcher);


/** @override */
goog.testing.mockmatchers.ObjectEquals.prototype.matches =
    function(toVerify, opt_expectation) {
  // Override the default matches implementation to capture the exception thrown
  // by assertObjectEquals (if any) and add that message to the expectation.
  try {
    return goog.testing.mockmatchers.ObjectEquals.superClass_.matches.call(
        this, toVerify, opt_expectation);
  } catch (e) {
    if (opt_expectation) {
      opt_expectation.addErrorMessage(e.message);
    }
    return false;
  }
};



/**
 * A matcher that saves the argument that it is verifying so that your unit test
 * can perform extra tests with this argument later.  For example, if the
 * argument is a callback method, the unit test can then later call this
 * callback to test the asynchronous portion of the call.
 * @param {goog.testing.mockmatchers.ArgumentMatcher|Function=} opt_matcher
 *     Argument matcher or matching function that will be used to validate the
 *     argument.  By default, argument will always be valid.
 * @param {?string=} opt_matchName The name expressing intent as part of
 *      an error message for when a match fails.
 * @constructor
 * @extends {goog.testing.mockmatchers.ArgumentMatcher}
 * @final
 */
goog.testing.mockmatchers.SaveArgument = function(opt_matcher, opt_matchName) {
  goog.testing.mockmatchers.ArgumentMatcher.call(
      this, /** @type {Function} */ (opt_matcher), opt_matchName);

  if (opt_matcher instanceof goog.testing.mockmatchers.ArgumentMatcher) {
    /**
     * Delegate match requests to this matcher.
     * @type {goog.testing.mockmatchers.ArgumentMatcher}
     * @private
     */
    this.delegateMatcher_ = opt_matcher;
  } else if (!opt_matcher) {
    this.delegateMatcher_ = goog.testing.mockmatchers.ignoreArgument;
  }
};
goog.inherits(goog.testing.mockmatchers.SaveArgument,
    goog.testing.mockmatchers.ArgumentMatcher);


/** @override */
goog.testing.mockmatchers.SaveArgument.prototype.matches = function(
    toVerify, opt_expectation) {
  this.arg = toVerify;
  if (this.delegateMatcher_) {
    return this.delegateMatcher_.matches(toVerify, opt_expectation);
  }
  return goog.testing.mockmatchers.SaveArgument.superClass_.matches.call(
      this, toVerify, opt_expectation);
};


/**
 * Saved argument that was verified.
 * @type {*}
 */
goog.testing.mockmatchers.SaveArgument.prototype.arg;


/**
 * An instance of the IgnoreArgument matcher. Returns true for all matches.
 * @type {goog.testing.mockmatchers.IgnoreArgument}
 */
goog.testing.mockmatchers.ignoreArgument =
    new goog.testing.mockmatchers.IgnoreArgument();


/**
 * A matcher that verifies that an argument is an array.
 * @type {goog.testing.mockmatchers.ArgumentMatcher}
 */
goog.testing.mockmatchers.isArray =
    new goog.testing.mockmatchers.ArgumentMatcher(goog.isArray,
        'isArray');


/**
 * A matcher that verifies that an argument is a array-like.  A NodeList is an
 * example of a collection that is very close to an array.
 * @type {goog.testing.mockmatchers.ArgumentMatcher}
 */
goog.testing.mockmatchers.isArrayLike =
    new goog.testing.mockmatchers.ArgumentMatcher(goog.isArrayLike,
        'isArrayLike');


/**
 * A matcher that verifies that an argument is a date-like.
 * @type {goog.testing.mockmatchers.ArgumentMatcher}
 */
goog.testing.mockmatchers.isDateLike =
    new goog.testing.mockmatchers.ArgumentMatcher(goog.isDateLike,
        'isDateLike');


/**
 * A matcher that verifies that an argument is a string.
 * @type {goog.testing.mockmatchers.ArgumentMatcher}
 */
goog.testing.mockmatchers.isString =
    new goog.testing.mockmatchers.ArgumentMatcher(goog.isString,
        'isString');


/**
 * A matcher that verifies that an argument is a boolean.
 * @type {goog.testing.mockmatchers.ArgumentMatcher}
 */
goog.testing.mockmatchers.isBoolean =
    new goog.testing.mockmatchers.ArgumentMatcher(goog.isBoolean,
        'isBoolean');


/**
 * A matcher that verifies that an argument is a number.
 * @type {goog.testing.mockmatchers.ArgumentMatcher}
 */
goog.testing.mockmatchers.isNumber =
    new goog.testing.mockmatchers.ArgumentMatcher(goog.isNumber,
        'isNumber');


/**
 * A matcher that verifies that an argument is a function.
 * @type {goog.testing.mockmatchers.ArgumentMatcher}
 */
goog.testing.mockmatchers.isFunction =
    new goog.testing.mockmatchers.ArgumentMatcher(goog.isFunction,
        'isFunction');


/**
 * A matcher that verifies that an argument is an object.
 * @type {goog.testing.mockmatchers.ArgumentMatcher}
 */
goog.testing.mockmatchers.isObject =
    new goog.testing.mockmatchers.ArgumentMatcher(goog.isObject,
        'isObject');


/**
 * A matcher that verifies that an argument is like a DOM node.
 * @type {goog.testing.mockmatchers.ArgumentMatcher}
 */
goog.testing.mockmatchers.isNodeLike =
    new goog.testing.mockmatchers.ArgumentMatcher(goog.dom.isNodeLike,
        'isNodeLike');


/**
 * A function that checks to see if an array matches a given set of
 * expectations.  The expectations array can be a mix of ArgumentMatcher
 * implementations and values.  True will be returned if values are identical or
 * if a matcher returns a positive result.
 * @param {Array} expectedArr An array of expectations which can be either
 *     values to check for equality or ArgumentMatchers.
 * @param {Array} arr The array to match.
 * @param {goog.testing.MockExpectation?=} opt_expectation The expectation
 *     for this match.
 * @return {boolean} Whether or not the given array matches the expectations.
 */
goog.testing.mockmatchers.flexibleArrayMatcher =
    function(expectedArr, arr, opt_expectation) {
  return goog.array.equals(expectedArr, arr, function(a, b) {
    var errCount = 0;
    if (opt_expectation) {
      errCount = opt_expectation.getErrorMessageCount();
    }
    var isamatch = a === b ||
        a instanceof goog.testing.mockmatchers.ArgumentMatcher &&
        a.matches(b, opt_expectation);
    var failureMessage = null;
    if (!isamatch) {
      failureMessage = goog.testing.asserts.findDifferences(a, b);
      isamatch = !failureMessage;
    }
    if (!isamatch && opt_expectation) {
      // If the error count changed, the match sent out an error
      // message. If the error count has not changed, then
      // we need to send out an error message...
      if (errCount == opt_expectation.getErrorMessageCount()) {
        // Use the _displayStringForValue() from assert.js
        // for consistency...
        if (!failureMessage) {
          failureMessage = 'Expected: ' + _displayStringForValue(a) +
              ' but was: ' + _displayStringForValue(b);
        }
        opt_expectation.addErrorMessage(failureMessage);
      }
    }
    return isamatch;
  });
};
