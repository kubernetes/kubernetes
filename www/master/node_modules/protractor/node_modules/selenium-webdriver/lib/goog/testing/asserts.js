// Copyright 2010 The Closure Library Authors. All Rights Reserved.
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
goog.provide('goog.testing.JsUnitException');
goog.provide('goog.testing.asserts');

goog.require('goog.testing.stacktrace');

// TODO(user): Copied from JsUnit with some small modifications, we should
// reimplement the asserters.


/**
 * @typedef {Array|NodeList|Arguments|{length: number}}
 */
goog.testing.asserts.ArrayLike;

var DOUBLE_EQUALITY_PREDICATE = function(var1, var2) {
  return var1 == var2;
};
var JSUNIT_UNDEFINED_VALUE;
var TO_STRING_EQUALITY_PREDICATE = function(var1, var2) {
  return var1.toString() === var2.toString();
};

var PRIMITIVE_EQUALITY_PREDICATES = {
  'String': DOUBLE_EQUALITY_PREDICATE,
  'Number': DOUBLE_EQUALITY_PREDICATE,
  'Boolean': DOUBLE_EQUALITY_PREDICATE,
  'Date': function(date1, date2) {
    return date1.getTime() == date2.getTime();
  },
  'RegExp': TO_STRING_EQUALITY_PREDICATE,
  'Function': TO_STRING_EQUALITY_PREDICATE
};


/**
 * Compares equality of two numbers, allowing them to differ up to a given
 * tolerance.
 * @param {number} var1 A number.
 * @param {number} var2 A number.
 * @param {number} tolerance the maximum allowed difference.
 * @return {boolean} Whether the two variables are sufficiently close.
 * @private
 */
goog.testing.asserts.numberRoughEqualityPredicate_ = function(
    var1, var2, tolerance) {
  return Math.abs(var1 - var2) <= tolerance;
};


/**
 * @type {Object.<string, function(*, *, number): boolean>}
 * @private
 */
goog.testing.asserts.primitiveRoughEqualityPredicates_ = {
  'Number': goog.testing.asserts.numberRoughEqualityPredicate_
};


var _trueTypeOf = function(something) {
  var result = typeof something;
  try {
    switch (result) {
      case 'string':
        break;
      case 'boolean':
        break;
      case 'number':
        break;
      case 'object':
        if (something == null) {
          result = 'null';
          break;
        }
      case 'function':
        switch (something.constructor) {
          case new String('').constructor:
            result = 'String';
            break;
          case new Boolean(true).constructor:
            result = 'Boolean';
            break;
          case new Number(0).constructor:
            result = 'Number';
            break;
          case new Array().constructor:
            result = 'Array';
            break;
          case new RegExp().constructor:
            result = 'RegExp';
            break;
          case new Date().constructor:
            result = 'Date';
            break;
          case Function:
            result = 'Function';
            break;
          default:
            var m = something.constructor.toString().match(
                /function\s*([^( ]+)\(/);
            if (m) {
              result = m[1];
            } else {
              break;
            }
        }
        break;
    }
  } catch (e) {

  } finally {
    result = result.substr(0, 1).toUpperCase() + result.substr(1);
  }
  return result;
};

var _displayStringForValue = function(aVar) {
  var result;
  try {
    result = '<' + String(aVar) + '>';
  } catch (ex) {
    result = '<toString failed: ' + ex.message + '>';
    // toString does not work on this object :-(
  }
  if (!(aVar === null || aVar === JSUNIT_UNDEFINED_VALUE)) {
    result += ' (' + _trueTypeOf(aVar) + ')';
  }
  return result;
};

var fail = function(failureMessage) {
  goog.testing.asserts.raiseException('Call to fail()', failureMessage);
};

var argumentsIncludeComments = function(expectedNumberOfNonCommentArgs, args) {
  return args.length == expectedNumberOfNonCommentArgs + 1;
};

var commentArg = function(expectedNumberOfNonCommentArgs, args) {
  if (argumentsIncludeComments(expectedNumberOfNonCommentArgs, args)) {
    return args[0];
  }

  return null;
};

var nonCommentArg = function(desiredNonCommentArgIndex,
    expectedNumberOfNonCommentArgs, args) {
  return argumentsIncludeComments(expectedNumberOfNonCommentArgs, args) ?
      args[desiredNonCommentArgIndex] :
      args[desiredNonCommentArgIndex - 1];
};

var _validateArguments = function(expectedNumberOfNonCommentArgs, args) {
  var valid = args.length == expectedNumberOfNonCommentArgs ||
      args.length == expectedNumberOfNonCommentArgs + 1 &&
      goog.isString(args[0]);
  _assert(null, valid, 'Incorrect arguments passed to assert function');
};

var _assert = function(comment, booleanValue, failureMessage) {
  if (!booleanValue) {
    goog.testing.asserts.raiseException(comment, failureMessage);
  }
};


/**
 * @param {*} expected The expected value.
 * @param {*} actual The actual value.
 * @return {string} A failure message of the values don't match.
 * @private
 */
goog.testing.asserts.getDefaultErrorMsg_ = function(expected, actual) {
  var msg = 'Expected ' + _displayStringForValue(expected) + ' but was ' +
      _displayStringForValue(actual);
  if ((typeof expected == 'string') && (typeof actual == 'string')) {
    // Try to find a human-readable difference.
    var limit = Math.min(expected.length, actual.length);
    var commonPrefix = 0;
    while (commonPrefix < limit &&
        expected.charAt(commonPrefix) == actual.charAt(commonPrefix)) {
      commonPrefix++;
    }

    var commonSuffix = 0;
    while (commonSuffix < limit &&
        expected.charAt(expected.length - commonSuffix - 1) ==
            actual.charAt(actual.length - commonSuffix - 1)) {
      commonSuffix++;
    }

    if (commonPrefix + commonSuffix > limit) {
      commonSuffix = 0;
    }

    if (commonPrefix > 2 || commonSuffix > 2) {
      var printString = function(str) {
        var startIndex = Math.max(0, commonPrefix - 2);
        var endIndex = Math.min(str.length, str.length - (commonSuffix - 2));
        return (startIndex > 0 ? '...' : '') +
            str.substring(startIndex, endIndex) +
            (endIndex < str.length ? '...' : '');
      };

      msg += '\nDifference was at position ' + commonPrefix +
          '. Expected [' + printString(expected) +
          '] vs. actual [' + printString(actual) + ']';
    }
  }
  return msg;
};


/**
 * @param {*} a The value to assert (1 arg) or debug message (2 args).
 * @param {*=} opt_b The value to assert (2 args only).
 */
var assert = function(a, opt_b) {
  _validateArguments(1, arguments);
  var comment = commentArg(1, arguments);
  var booleanValue = nonCommentArg(1, 1, arguments);

  _assert(comment, goog.isBoolean(booleanValue),
      'Bad argument to assert(boolean)');
  _assert(comment, booleanValue, 'Call to assert(boolean) with false');
};


/**
 * Asserts that the function throws an error.
 *
 * @param {!(string|Function)} a The assertion comment or the function to call.
 * @param {!Function=} opt_b The function to call (if the first argument of
 *     {@code assertThrows} was the comment).
 * @return {*} The error thrown by the function.
 * @throws {goog.testing.JsUnitException} If the assertion failed.
 */
var assertThrows = function(a, opt_b) {
  _validateArguments(1, arguments);
  var func = nonCommentArg(1, 1, arguments);
  var comment = commentArg(1, arguments);
  _assert(comment, typeof func == 'function',
      'Argument passed to assertThrows is not a function');

  try {
    func();
  } catch (e) {
    if (e && goog.isString(e['stacktrace']) && goog.isString(e['message'])) {
      // Remove the stack trace appended to the error message by Opera 10.0
      var startIndex = e['message'].length - e['stacktrace'].length;
      if (e['message'].indexOf(e['stacktrace'], startIndex) == startIndex) {
        e['message'] = e['message'].substr(0, startIndex - 14);
      }
    }
    return e;
  }
  goog.testing.asserts.raiseException(comment,
      'No exception thrown from function passed to assertThrows');
};


/**
 * Asserts that the function does not throw an error.
 *
 * @param {!(string|Function)} a The assertion comment or the function to call.
 * @param {!Function=} opt_b The function to call (if the first argument of
 *     {@code assertNotThrows} was the comment).
 * @return {*} The return value of the function.
 * @throws {goog.testing.JsUnitException} If the assertion failed.
 */
var assertNotThrows = function(a, opt_b) {
  _validateArguments(1, arguments);
  var comment = commentArg(1, arguments);
  var func = nonCommentArg(1, 1, arguments);
  _assert(comment, typeof func == 'function',
      'Argument passed to assertNotThrows is not a function');

  try {
    return func();
  } catch (e) {
    comment = comment ? (comment + '\n') : '';
    comment += 'A non expected exception was thrown from function passed to ' +
               'assertNotThrows';
    // Some browsers don't have a stack trace so at least have the error
    // description.
    var stackTrace = e['stack'] || e['stacktrace'] || e.toString();
    goog.testing.asserts.raiseException(comment, stackTrace);
  }
};


/**
 * Asserts that the given callback function results in a JsUnitException when
 * called, and that the resulting failure message matches the given expected
 * message.
 * @param {function() : void} callback Function to be run expected to result
 *     in a JsUnitException (usually contains a call to an assert).
 * @param {string=} opt_expectedMessage Failure message expected to be given
 *     with the exception.
 */
var assertThrowsJsUnitException = function(callback, opt_expectedMessage) {
  var failed = false;
  try {
    goog.testing.asserts.callWithoutLogging(callback);
  } catch (ex) {
    if (!ex.isJsUnitException) {
      fail('Expected a JsUnitException');
    }
    if (typeof opt_expectedMessage != 'undefined' &&
        ex.message != opt_expectedMessage) {
      fail('Expected message [' + opt_expectedMessage + '] but got [' +
          ex.message + ']');
    }
    failed = true;
  }
  if (!failed) {
    fail('Expected a failure: ' + opt_expectedMessage);
  }
};


/**
 * @param {*} a The value to assert (1 arg) or debug message (2 args).
 * @param {*=} opt_b The value to assert (2 args only).
 */
var assertTrue = function(a, opt_b) {
  _validateArguments(1, arguments);
  var comment = commentArg(1, arguments);
  var booleanValue = nonCommentArg(1, 1, arguments);

  _assert(comment, goog.isBoolean(booleanValue),
      'Bad argument to assertTrue(boolean)');
  _assert(comment, booleanValue, 'Call to assertTrue(boolean) with false');
};


/**
 * @param {*} a The value to assert (1 arg) or debug message (2 args).
 * @param {*=} opt_b The value to assert (2 args only).
 */
var assertFalse = function(a, opt_b) {
  _validateArguments(1, arguments);
  var comment = commentArg(1, arguments);
  var booleanValue = nonCommentArg(1, 1, arguments);

  _assert(comment, goog.isBoolean(booleanValue),
      'Bad argument to assertFalse(boolean)');
  _assert(comment, !booleanValue, 'Call to assertFalse(boolean) with true');
};


/**
 * @param {*} a The expected value (2 args) or the debug message (3 args).
 * @param {*} b The actual value (2 args) or the expected value (3 args).
 * @param {*=} opt_c The actual value (3 args only).
 */
var assertEquals = function(a, b, opt_c) {
  _validateArguments(2, arguments);
  var var1 = nonCommentArg(1, 2, arguments);
  var var2 = nonCommentArg(2, 2, arguments);
  _assert(commentArg(2, arguments), var1 === var2,
          goog.testing.asserts.getDefaultErrorMsg_(var1, var2));
};


/**
 * @param {*} a The expected value (2 args) or the debug message (3 args).
 * @param {*} b The actual value (2 args) or the expected value (3 args).
 * @param {*=} opt_c The actual value (3 args only).
 */
var assertNotEquals = function(a, b, opt_c) {
  _validateArguments(2, arguments);
  var var1 = nonCommentArg(1, 2, arguments);
  var var2 = nonCommentArg(2, 2, arguments);
  _assert(commentArg(2, arguments), var1 !== var2,
      'Expected not to be ' + _displayStringForValue(var2));
};


/**
 * @param {*} a The value to assert (1 arg) or debug message (2 args).
 * @param {*=} opt_b The value to assert (2 args only).
 */
var assertNull = function(a, opt_b) {
  _validateArguments(1, arguments);
  var aVar = nonCommentArg(1, 1, arguments);
  _assert(commentArg(1, arguments), aVar === null,
      goog.testing.asserts.getDefaultErrorMsg_(null, aVar));
};


/**
 * @param {*} a The value to assert (1 arg) or debug message (2 args).
 * @param {*=} opt_b The value to assert (2 args only).
 */
var assertNotNull = function(a, opt_b) {
  _validateArguments(1, arguments);
  var aVar = nonCommentArg(1, 1, arguments);
  _assert(commentArg(1, arguments), aVar !== null,
      'Expected not to be ' + _displayStringForValue(null));
};


/**
 * @param {*} a The value to assert (1 arg) or debug message (2 args).
 * @param {*=} opt_b The value to assert (2 args only).
 */
var assertUndefined = function(a, opt_b) {
  _validateArguments(1, arguments);
  var aVar = nonCommentArg(1, 1, arguments);
  _assert(commentArg(1, arguments), aVar === JSUNIT_UNDEFINED_VALUE,
      goog.testing.asserts.getDefaultErrorMsg_(JSUNIT_UNDEFINED_VALUE, aVar));
};


/**
 * @param {*} a The value to assert (1 arg) or debug message (2 args).
 * @param {*=} opt_b The value to assert (2 args only).
 */
var assertNotUndefined = function(a, opt_b) {
  _validateArguments(1, arguments);
  var aVar = nonCommentArg(1, 1, arguments);
  _assert(commentArg(1, arguments), aVar !== JSUNIT_UNDEFINED_VALUE,
      'Expected not to be ' + _displayStringForValue(JSUNIT_UNDEFINED_VALUE));
};


/**
 * @param {*} a The value to assert (1 arg) or debug message (2 args).
 * @param {*=} opt_b The value to assert (2 args only).
 */
var assertNotNullNorUndefined = function(a, opt_b) {
  _validateArguments(1, arguments);
  assertNotNull.apply(null, arguments);
  assertNotUndefined.apply(null, arguments);
};


/**
 * @param {*} a The value to assert (1 arg) or debug message (2 args).
 * @param {*=} opt_b The value to assert (2 args only).
 */
var assertNonEmptyString = function(a, opt_b) {
  _validateArguments(1, arguments);
  var aVar = nonCommentArg(1, 1, arguments);
  _assert(commentArg(1, arguments),
      aVar !== JSUNIT_UNDEFINED_VALUE && aVar !== null &&
      typeof aVar == 'string' && aVar !== '',
      'Expected non-empty string but was ' + _displayStringForValue(aVar));
};


/**
 * @param {*} a The value to assert (1 arg) or debug message (2 args).
 * @param {*=} opt_b The value to assert (2 args only).
 */
var assertNaN = function(a, opt_b) {
  _validateArguments(1, arguments);
  var aVar = nonCommentArg(1, 1, arguments);
  _assert(commentArg(1, arguments), isNaN(aVar), 'Expected NaN');
};


/**
 * @param {*} a The value to assert (1 arg) or debug message (2 args).
 * @param {*=} opt_b The value to assert (2 args only).
 */
var assertNotNaN = function(a, opt_b) {
  _validateArguments(1, arguments);
  var aVar = nonCommentArg(1, 1, arguments);
  _assert(commentArg(1, arguments), !isNaN(aVar), 'Expected not NaN');
};


/**
 * Runs a function in an environment where test failures are not logged. This is
 * useful for testing test code, where failures can be a normal part of a test.
 * @param {function() : void} fn Function to run without logging failures.
 */
goog.testing.asserts.callWithoutLogging = function(fn) {
  var testRunner = goog.global['G_testRunner'];
  var oldLogTestFailure = testRunner['logTestFailure'];
  try {
    // Any failures in the callback shouldn't be recorded.
    testRunner['logTestFailure'] = undefined;
    fn();
  } finally {
    testRunner['logTestFailure'] = oldLogTestFailure;
  }
};


/**
 * The return value of the equality predicate passed to findDifferences below,
 * in cases where the predicate can't test the input variables for equality.
 * @type {?string}
 */
goog.testing.asserts.EQUALITY_PREDICATE_CANT_PROCESS = null;


/**
 * The return value of the equality predicate passed to findDifferences below,
 * in cases where the input vriables are equal.
 * @type {?string}
 */
goog.testing.asserts.EQUALITY_PREDICATE_VARS_ARE_EQUAL = '';


/**
 * Determines if two items of any type match, and formulates an error message
 * if not.
 * @param {*} expected Expected argument to match.
 * @param {*} actual Argument as a result of performing the test.
 * @param {(function(string, *, *): ?string)=} opt_equalityPredicate An optional
 *     function that can be used to check equality of variables. It accepts 3
 *     arguments: type-of-variables, var1, var2 (in that order) and returns an
 *     error message if the variables are not equal,
 *     goog.testing.asserts.EQUALITY_PREDICATE_VARS_ARE_EQUAL if the variables
 *     are equal, or
 *     goog.testing.asserts.EQUALITY_PREDICATE_CANT_PROCESS if the predicate
 *     couldn't check the input variables. The function will be called only if
 *     the types of var1 and var2 are identical.
 * @return {?string} Null on success, error message on failure.
 */
goog.testing.asserts.findDifferences = function(expected, actual,
    opt_equalityPredicate) {
  var failures = [];
  var seen1 = [];
  var seen2 = [];

  // To avoid infinite recursion when the two parameters are self-referential
  // along the same path of properties, keep track of the object pairs already
  // seen in this call subtree, and abort when a cycle is detected.
  function innerAssert(var1, var2, path) {
    // This is used for testing, so we can afford to be slow (but more
    // accurate). So we just check whether var1 is in seen1. If we
    // found var1 in index i, we simply need to check whether var2 is
    // in seen2[i]. If it is, we do not recurse to check var1/var2. If
    // it isn't, we know that the structures of the two objects must be
    // different.
    //
    // This is based on the fact that values at index i in seen1 and
    // seen2 will be checked for equality eventually (when
    // innerAssert_(seen1[i], seen2[i], path) finishes).
    for (var i = 0; i < seen1.length; ++i) {
      var match1 = seen1[i] === var1;
      var match2 = seen2[i] === var2;
      if (match1 || match2) {
        if (!match1 || !match2) {
          // Asymmetric cycles, so the objects have different structure.
          failures.push('Asymmetric cycle detected at ' + path);
        }
        return;
      }
    }

    seen1.push(var1);
    seen2.push(var2);
    innerAssert_(var1, var2, path);
    seen1.pop();
    seen2.pop();
  }

  var equalityPredicate = opt_equalityPredicate || function(type, var1, var2) {
    var typedPredicate = PRIMITIVE_EQUALITY_PREDICATES[type];
    if (!typedPredicate) {
      return goog.testing.asserts.EQUALITY_PREDICATE_CANT_PROCESS;
    }
    var equal = typedPredicate(var1, var2);
    return equal ? goog.testing.asserts.EQUALITY_PREDICATE_VARS_ARE_EQUAL :
        goog.testing.asserts.getDefaultErrorMsg_(var1, var2);
  };

  /**
   * @param {*} var1 An item in the expected object.
   * @param {*} var2 The corresponding item in the actual object.
   * @param {string} path Their path in the objects.
   * @suppress {missingProperties} The map_ property is unknown to the compiler
   *     unless goog.structs.Map is loaded.
   */
  function innerAssert_(var1, var2, path) {
    if (var1 === var2) {
      return;
    }

    var typeOfVar1 = _trueTypeOf(var1);
    var typeOfVar2 = _trueTypeOf(var2);

    if (typeOfVar1 == typeOfVar2) {
      var isArray = typeOfVar1 == 'Array';
      var errorMessage = equalityPredicate(typeOfVar1, var1, var2);
      if (errorMessage !=
          goog.testing.asserts.EQUALITY_PREDICATE_CANT_PROCESS) {
        if (errorMessage !=
            goog.testing.asserts.EQUALITY_PREDICATE_VARS_ARE_EQUAL) {
          failures.push(path + ': ' + errorMessage);
        }
      } else if (isArray && var1.length != var2.length) {
        failures.push(path + ': Expected ' + var1.length + '-element array ' +
                      'but got a ' + var2.length + '-element array');
      } else {
        var childPath = path + (isArray ? '[%s]' : (path ? '.%s' : '%s'));

        // if an object has an __iterator__ property, we have no way of
        // actually inspecting its raw properties, and JS 1.7 doesn't
        // overload [] to make it possible for someone to generically
        // use what the iterator returns to compare the object-managed
        // properties. This gets us into deep poo with things like
        // goog.structs.Map, at least on systems that support iteration.
        if (!var1['__iterator__']) {
          for (var prop in var1) {
            if (isArray && goog.testing.asserts.isArrayIndexProp_(prop)) {
              // Skip array indices for now. We'll handle them later.
              continue;
            }

            if (prop in var2) {
              innerAssert(var1[prop], var2[prop],
                          childPath.replace('%s', prop));
            } else {
              failures.push('property ' + prop +
                            ' not present in actual ' + (path || typeOfVar2));
            }
          }
          // make sure there aren't properties in var2 that are missing
          // from var1. if there are, then by definition they don't
          // match.
          for (var prop in var2) {
            if (isArray && goog.testing.asserts.isArrayIndexProp_(prop)) {
              // Skip array indices for now. We'll handle them later.
              continue;
            }

            if (!(prop in var1)) {
              failures.push('property ' + prop +
                            ' not present in expected ' +
                            (path || typeOfVar1));
            }
          }

          // Handle array indices by iterating from 0 to arr.length.
          //
          // Although all browsers allow holes in arrays, browsers
          // are inconsistent in what they consider a hole. For example,
          // "[0,undefined,2]" has a hole on IE but not on Firefox.
          //
          // Because our style guide bans for...in iteration over arrays,
          // we assume that most users don't care about holes in arrays,
          // and that it is ok to say that a hole is equivalent to a slot
          // populated with 'undefined'.
          if (isArray) {
            for (prop = 0; prop < var1.length; prop++) {
              innerAssert(var1[prop], var2[prop],
                          childPath.replace('%s', String(prop)));
            }
          }
        } else {
          // special-case for closure objects that have iterators
          if (goog.isFunction(var1.equals)) {
            // use the object's own equals function, assuming it accepts an
            // object and returns a boolean
            if (!var1.equals(var2)) {
              failures.push('equals() returned false for ' +
                            (path || typeOfVar1));
            }
          } else if (var1.map_) {
            // assume goog.structs.Map or goog.structs.Set, where comparing
            // their private map_ field is sufficient
            innerAssert(var1.map_, var2.map_, childPath.replace('%s', 'map_'));
          } else {
            // else die, so user knows we can't do anything
            failures.push('unable to check ' + (path || typeOfVar1) +
                          ' for equality: it has an iterator we do not ' +
                          'know how to handle. please add an equals method');
          }
        }
      }
    } else {
      failures.push(path + ' ' +
          goog.testing.asserts.getDefaultErrorMsg_(var1, var2));
    }
  }

  innerAssert(expected, actual, '');
  return failures.length == 0 ? null :
      goog.testing.asserts.getDefaultErrorMsg_(expected, actual) +
          '\n   ' + failures.join('\n   ');
};


/**
 * Notes:
 * Object equality has some nasty browser quirks, and this implementation is
 * not 100% correct. For example,
 *
 * <code>
 * var a = [0, 1, 2];
 * var b = [0, 1, 2];
 * delete a[1];
 * b[1] = undefined;
 * assertObjectEquals(a, b); // should fail, but currently passes
 * </code>
 *
 * See asserts_test.html for more interesting edge cases.
 *
 * The first comparison object provided is the expected value, the second is
 * the actual.
 *
 * @param {*} a Assertion message or comparison object.
 * @param {*} b Comparison object.
 * @param {*=} opt_c Comparison object, if an assertion message was provided.
 */
var assertObjectEquals = function(a, b, opt_c) {
  _validateArguments(2, arguments);
  var v1 = nonCommentArg(1, 2, arguments);
  var v2 = nonCommentArg(2, 2, arguments);
  var failureMessage = commentArg(2, arguments) ? commentArg(2, arguments) : '';
  var differences = goog.testing.asserts.findDifferences(v1, v2);

  _assert(failureMessage, !differences, differences);
};


/**
 * Similar to assertObjectEquals above, but accepts a tolerance margin.
 *
 * @param {*} a Assertion message or comparison object.
 * @param {*} b Comparison object.
 * @param {*} c Comparison object or tolerance.
 * @param {*=} opt_d Tolerance, if an assertion message was provided.
 */
var assertObjectRoughlyEquals = function(a, b, c, opt_d) {
  _validateArguments(3, arguments);
  var v1 = nonCommentArg(1, 3, arguments);
  var v2 = nonCommentArg(2, 3, arguments);
  var tolerance = nonCommentArg(3, 3, arguments);
  var failureMessage = commentArg(3, arguments) ? commentArg(3, arguments) : '';
  var equalityPredicate = function(type, var1, var2) {
    var typedPredicate =
        goog.testing.asserts.primitiveRoughEqualityPredicates_[type];
    if (!typedPredicate) {
      return goog.testing.asserts.EQUALITY_PREDICATE_CANT_PROCESS;
    }
    var equal = typedPredicate(var1, var2, tolerance);
    return equal ? goog.testing.asserts.EQUALITY_PREDICATE_VARS_ARE_EQUAL :
        goog.testing.asserts.getDefaultErrorMsg_(var1, var2) +
        ' which was more than ' + tolerance + ' away';
  };
  var differences = goog.testing.asserts.findDifferences(
      v1, v2, equalityPredicate);

  _assert(failureMessage, !differences, differences);
};


/**
 * Compares two arbitrary objects for non-equalness.
 *
 * All the same caveats as for assertObjectEquals apply here:
 * Undefined values may be confused for missing values, or vice versa.
 *
 * @param {*} a Assertion message or comparison object.
 * @param {*} b Comparison object.
 * @param {*=} opt_c Comparison object, if an assertion message was provided.
 */
var assertObjectNotEquals = function(a, b, opt_c) {
  _validateArguments(2, arguments);
  var v1 = nonCommentArg(1, 2, arguments);
  var v2 = nonCommentArg(2, 2, arguments);
  var failureMessage = commentArg(2, arguments) ? commentArg(2, arguments) : '';
  var differences = goog.testing.asserts.findDifferences(v1, v2);

  _assert(failureMessage, differences, 'Objects should not be equal');
};


/**
 * Compares two arrays ignoring negative indexes and extra properties on the
 * array objects. Use case: Internet Explorer adds the index, lastIndex and
 * input enumerable fields to the result of string.match(/regexp/g), which makes
 * assertObjectEquals fail.
 * @param {*} a The expected array (2 args) or the debug message (3 args).
 * @param {*} b The actual array (2 args) or the expected array (3 args).
 * @param {*=} opt_c The actual array (3 args only).
 */
var assertArrayEquals = function(a, b, opt_c) {
  _validateArguments(2, arguments);
  var v1 = nonCommentArg(1, 2, arguments);
  var v2 = nonCommentArg(2, 2, arguments);
  var failureMessage = commentArg(2, arguments) ? commentArg(2, arguments) : '';

  var typeOfVar1 = _trueTypeOf(v1);
  _assert(failureMessage,
          typeOfVar1 == 'Array',
          'Expected an array for assertArrayEquals but found a ' + typeOfVar1);

  var typeOfVar2 = _trueTypeOf(v2);
  _assert(failureMessage,
          typeOfVar2 == 'Array',
          'Expected an array for assertArrayEquals but found a ' + typeOfVar2);

  assertObjectEquals(failureMessage,
      Array.prototype.concat.call(v1), Array.prototype.concat.call(v2));
};


/**
 * Compares two objects that can be accessed like an array and assert that
 * each element is equal.
 * @param {string|Object} a Failure message (3 arguments)
 *     or object #1 (2 arguments).
 * @param {Object} b Object #1 (2 arguments) or object #2 (3 arguments).
 * @param {Object=} opt_c Object #2 (3 arguments).
 */
var assertElementsEquals = function(a, b, opt_c) {
  _validateArguments(2, arguments);

  var v1 = nonCommentArg(1, 2, arguments);
  var v2 = nonCommentArg(2, 2, arguments);
  var failureMessage = commentArg(2, arguments) ? commentArg(2, arguments) : '';

  if (!v1) {
    assert(failureMessage, !v2);
  } else {
    assertEquals('length mismatch: ' + failureMessage, v1.length, v2.length);
    for (var i = 0; i < v1.length; ++i) {
      assertEquals(
          'mismatch at index ' + i + ': ' + failureMessage, v1[i], v2[i]);
    }
  }
};


/**
 * Compares two objects that can be accessed like an array and assert that
 * each element is roughly equal.
 * @param {string|Object} a Failure message (4 arguments)
 *     or object #1 (3 arguments).
 * @param {Object} b Object #1 (4 arguments) or object #2 (3 arguments).
 * @param {Object|number} c Object #2 (4 arguments) or tolerance (3 arguments).
 * @param {number=} opt_d tolerance (4 arguments).
 */
var assertElementsRoughlyEqual = function(a, b, c, opt_d) {
  _validateArguments(3, arguments);

  var v1 = nonCommentArg(1, 3, arguments);
  var v2 = nonCommentArg(2, 3, arguments);
  var tolerance = nonCommentArg(3, 3, arguments);
  var failureMessage = commentArg(3, arguments) ? commentArg(3, arguments) : '';

  if (!v1) {
    assert(failureMessage, !v2);
  } else {
    assertEquals('length mismatch: ' + failureMessage, v1.length, v2.length);
    for (var i = 0; i < v1.length; ++i) {
      assertRoughlyEquals(failureMessage, v1[i], v2[i], tolerance);
    }
  }
};


/**
 * Compares two array-like objects without taking their order into account.
 * @param {string|goog.testing.asserts.ArrayLike} a Assertion message or the
 *     expected elements.
 * @param {goog.testing.asserts.ArrayLike} b Expected elements or the actual
 *     elements.
 * @param {goog.testing.asserts.ArrayLike=} opt_c Actual elements.
 */
var assertSameElements = function(a, b, opt_c) {
  _validateArguments(2, arguments);
  var expected = nonCommentArg(1, 2, arguments);
  var actual = nonCommentArg(2, 2, arguments);
  var message = commentArg(2, arguments);

  assertTrue('Bad arguments to assertSameElements(opt_message, expected: ' +
      'ArrayLike, actual: ArrayLike)',
      goog.isArrayLike(expected) && goog.isArrayLike(actual));

  // Clones expected and actual and converts them to real arrays.
  expected = goog.testing.asserts.toArray_(expected);
  actual = goog.testing.asserts.toArray_(actual);
  // TODO(user): It would be great to show only the difference
  // between the expected and actual elements.
  _assert(message, expected.length == actual.length,
      'Expected ' + expected.length + ' elements: [' + expected + '], ' +
      'got ' + actual.length + ' elements: [' + actual + ']');

  var toFind = goog.testing.asserts.toArray_(expected);
  for (var i = 0; i < actual.length; i++) {
    var index = goog.testing.asserts.indexOf_(toFind, actual[i]);
    _assert(message, index != -1, 'Expected [' + expected + '], got [' +
        actual + ']');
    toFind.splice(index, 1);
  }
};


/**
 * @param {*} a The value to assert (1 arg) or debug message (2 args).
 * @param {*=} opt_b The value to assert (2 args only).
 */
var assertEvaluatesToTrue = function(a, opt_b) {
  _validateArguments(1, arguments);
  var value = nonCommentArg(1, 1, arguments);
  if (!value) {
    _assert(commentArg(1, arguments), false, 'Expected to evaluate to true');
  }
};


/**
 * @param {*} a The value to assert (1 arg) or debug message (2 args).
 * @param {*=} opt_b The value to assert (2 args only).
 */
var assertEvaluatesToFalse = function(a, opt_b) {
  _validateArguments(1, arguments);
  var value = nonCommentArg(1, 1, arguments);
  if (value) {
    _assert(commentArg(1, arguments), false, 'Expected to evaluate to false');
  }
};


/**
 * Compares two HTML snippets.
 *
 * Take extra care if attributes are involved. {@code assertHTMLEquals}'s
 * implementation isn't prepared for complex cases. For example, the following
 * comparisons erroneously fail:
 * <pre>
 * assertHTMLEquals('<a href="x" target="y">', '<a target="y" href="x">');
 * assertHTMLEquals('<div classname="a b">', '<div classname="b a">');
 * assertHTMLEquals('<input disabled>', '<input disabled="disabled">');
 * </pre>
 *
 * When in doubt, use {@code goog.testing.dom.assertHtmlMatches}.
 *
 * @param {*} a The expected value (2 args) or the debug message (3 args).
 * @param {*} b The actual value (2 args) or the expected value (3 args).
 * @param {*=} opt_c The actual value (3 args only).
 */
var assertHTMLEquals = function(a, b, opt_c) {
  _validateArguments(2, arguments);
  var var1 = nonCommentArg(1, 2, arguments);
  var var2 = nonCommentArg(2, 2, arguments);
  var var1Standardized = standardizeHTML(var1);
  var var2Standardized = standardizeHTML(var2);

  _assert(commentArg(2, arguments), var1Standardized === var2Standardized,
          goog.testing.asserts.getDefaultErrorMsg_(
              var1Standardized, var2Standardized));
};


/**
 * Compares two CSS property values to make sure that they represent the same
 * things. This will normalize values in the browser. For example, in Firefox,
 * this assertion will consider "rgb(0, 0, 255)" and "#0000ff" to be identical
 * values for the "color" property. This function won't normalize everything --
 * for example, in most browsers, "blue" will not match "#0000ff". It is
 * intended only to compensate for unexpected normalizations performed by
 * the browser that should also affect your expected value.
 * @param {string} a Assertion message, or the CSS property name.
 * @param {string} b CSS property name, or the expected value.
 * @param {string} c The expected value, or the actual value.
 * @param {string=} opt_d The actual value.
 */
var assertCSSValueEquals = function(a, b, c, opt_d) {
  _validateArguments(3, arguments);
  var propertyName = nonCommentArg(1, 3, arguments);
  var expectedValue = nonCommentArg(2, 3, arguments);
  var actualValue = nonCommentArg(3, 3, arguments);
  var expectedValueStandardized =
      standardizeCSSValue(propertyName, expectedValue);
  var actualValueStandardized =
      standardizeCSSValue(propertyName, actualValue);

  _assert(commentArg(3, arguments),
          expectedValueStandardized == actualValueStandardized,
          goog.testing.asserts.getDefaultErrorMsg_(
              expectedValueStandardized, actualValueStandardized));
};


/**
 * @param {*} a The expected value (2 args) or the debug message (3 args).
 * @param {*} b The actual value (2 args) or the expected value (3 args).
 * @param {*=} opt_c The actual value (3 args only).
 */
var assertHashEquals = function(a, b, opt_c) {
  _validateArguments(2, arguments);
  var var1 = nonCommentArg(1, 2, arguments);
  var var2 = nonCommentArg(2, 2, arguments);
  var message = commentArg(2, arguments);
  for (var key in var1) {
    _assert(message,
        key in var2, 'Expected hash had key ' + key + ' that was not found');
    _assert(message, var1[key] == var2[key], 'Value for key ' + key +
        ' mismatch - expected = ' + var1[key] + ', actual = ' + var2[key]);
  }

  for (var key in var2) {
    _assert(message, key in var1, 'Actual hash had key ' + key +
        ' that was not expected');
  }
};


/**
 * @param {*} a The expected value (3 args) or the debug message (4 args).
 * @param {*} b The actual value (3 args) or the expected value (4 args).
 * @param {*} c The tolerance (3 args) or the actual value (4 args).
 * @param {*=} opt_d The tolerance (4 args only).
 */
var assertRoughlyEquals = function(a, b, c, opt_d) {
  _validateArguments(3, arguments);
  var expected = nonCommentArg(1, 3, arguments);
  var actual = nonCommentArg(2, 3, arguments);
  var tolerance = nonCommentArg(3, 3, arguments);
  _assert(commentArg(3, arguments),
      goog.testing.asserts.numberRoughEqualityPredicate_(
          expected, actual, tolerance),
      'Expected ' + expected + ', but got ' + actual +
      ' which was more than ' + tolerance + ' away');
};


/**
 * Checks if the given element is the member of the given container.
 * @param {*} a Failure message (3 arguments) or the contained element
 *     (2 arguments).
 * @param {*} b The contained element (3 arguments) or the container
 *     (2 arguments).
 * @param {*=} opt_c The container.
 */
var assertContains = function(a, b, opt_c) {
  _validateArguments(2, arguments);
  var contained = nonCommentArg(1, 2, arguments);
  var container = nonCommentArg(2, 2, arguments);
  _assert(commentArg(2, arguments),
      goog.testing.asserts.contains_(container, contained),
      'Expected \'' + container + '\' to contain \'' + contained + '\'');
};


/**
 * Checks if the given element is not the member of the given container.
 * @param {*} a Failure message (3 arguments) or the contained element
 *     (2 arguments).
 * @param {*} b The contained element (3 arguments) or the container
 *     (2 arguments).
 * @param {*=} opt_c The container.
 */
var assertNotContains = function(a, b, opt_c) {
  _validateArguments(2, arguments);
  var contained = nonCommentArg(1, 2, arguments);
  var container = nonCommentArg(2, 2, arguments);
  _assert(commentArg(2, arguments),
      !goog.testing.asserts.contains_(container, contained),
      'Expected \'' + container + '\' not to contain \'' + contained + '\'');
};


/**
 * Checks if the given string matches the given regular expression.
 * @param {*} a Failure message (3 arguments) or the expected regular
 *     expression as a string or RegExp (2 arguments).
 * @param {*} b The regular expression (3 arguments) or the string to test
 *     (2 arguments).
 * @param {*=} opt_c The string to test.
 */
var assertRegExp = function(a, b, opt_c) {
  _validateArguments(2, arguments);
  var regexp = nonCommentArg(1, 2, arguments);
  var string = nonCommentArg(2, 2, arguments);
  if (typeof(regexp) == 'string') {
    regexp = new RegExp(regexp);
  }
  _assert(commentArg(2, arguments),
      regexp.test(string),
      'Expected \'' + string + '\' to match RegExp ' + regexp.toString());
};


/**
 * Converts an array like object to array or clones it if it's already array.
 * @param {goog.testing.asserts.ArrayLike} arrayLike The collection.
 * @return {!Array} Copy of the collection as array.
 * @private
 */
goog.testing.asserts.toArray_ = function(arrayLike) {
  var ret = [];
  for (var i = 0; i < arrayLike.length; i++) {
    ret[i] = arrayLike[i];
  }
  return ret;
};


/**
 * Finds the position of the first occurrence of an element in a container.
 * @param {goog.testing.asserts.ArrayLike} container
 *     The array to find the element in.
 * @param {*} contained Element to find.
 * @return {number} Index of the first occurrence or -1 if not found.
 * @private
 */
goog.testing.asserts.indexOf_ = function(container, contained) {
  if (container.indexOf) {
    return container.indexOf(contained);
  } else {
    // IE6/7 do not have indexOf so do a search.
    for (var i = 0; i < container.length; i++) {
      if (container[i] === contained) {
        return i;
      }
    }
    return -1;
  }
};


/**
 * Tells whether the array contains the given element.
 * @param {goog.testing.asserts.ArrayLike} container The array to
 *     find the element in.
 * @param {*} contained Element to find.
 * @return {boolean} Whether the element is in the array.
 * @private
 */
goog.testing.asserts.contains_ = function(container, contained) {
  // TODO(user): Can we check for container.contains as well?
  // That would give us support for most goog.structs (though weird results
  // with anything else with a contains method, like goog.math.Range). Falling
  // back with container.some would catch all iterables, too.
  return goog.testing.asserts.indexOf_(container, contained) != -1;
};

var standardizeHTML = function(html) {
  var translator = document.createElement('DIV');
  translator.innerHTML = html;

  // Trim whitespace from result (without relying on goog.string)
  return translator.innerHTML.replace(/^\s+|\s+$/g, '');
};


/**
 * Standardizes a CSS value for a given property by applying it to an element
 * and then reading it back.
 * @param {string} propertyName CSS property name.
 * @param {string} value CSS value.
 * @return {string} Normalized CSS value.
 */
var standardizeCSSValue = function(propertyName, value) {
  var styleDeclaration = document.createElement('DIV').style;
  styleDeclaration[propertyName] = value;
  return styleDeclaration[propertyName];
};


/**
 * Raises a JsUnit exception with the given comment.
 * @param {string} comment A summary for the exception.
 * @param {string=} opt_message A description of the exception.
 */
goog.testing.asserts.raiseException = function(comment, opt_message) {
  throw new goog.testing.JsUnitException(comment, opt_message);
};


/**
 * Helper function for assertObjectEquals.
 * @param {string} prop A property name.
 * @return {boolean} If the property name is an array index.
 * @private
 */
goog.testing.asserts.isArrayIndexProp_ = function(prop) {
  return (prop | 0) == prop;
};



/**
 * @param {string} comment A summary for the exception.
 * @param {?string=} opt_message A description of the exception.
 * @constructor
 * @extends {Error}
 * @final
 */
goog.testing.JsUnitException = function(comment, opt_message) {
  this.isJsUnitException = true;
  this.message = (comment ? comment : '') +
                 (comment && opt_message ? '\n' : '') +
                 (opt_message ? opt_message : '');
  this.stackTrace = goog.testing.stacktrace.get();
  // These fields are for compatibility with jsUnitTestManager.
  this.comment = comment || null;
  this.jsUnitMessage = opt_message || '';

  // Ensure there is a stack trace.
  if (Error.captureStackTrace) {
    Error.captureStackTrace(this, goog.testing.JsUnitException);
  } else {
    this.stack = new Error().stack || '';
  }
};
goog.inherits(goog.testing.JsUnitException, Error);


/** @override */
goog.testing.JsUnitException.prototype.toString = function() {
  return this.message;
};


goog.exportSymbol('fail', fail);
goog.exportSymbol('assert', assert);
goog.exportSymbol('assertThrows', assertThrows);
goog.exportSymbol('assertNotThrows', assertNotThrows);
goog.exportSymbol('assertTrue', assertTrue);
goog.exportSymbol('assertFalse', assertFalse);
goog.exportSymbol('assertEquals', assertEquals);
goog.exportSymbol('assertNotEquals', assertNotEquals);
goog.exportSymbol('assertNull', assertNull);
goog.exportSymbol('assertNotNull', assertNotNull);
goog.exportSymbol('assertUndefined', assertUndefined);
goog.exportSymbol('assertNotUndefined', assertNotUndefined);
goog.exportSymbol('assertNotNullNorUndefined', assertNotNullNorUndefined);
goog.exportSymbol('assertNonEmptyString', assertNonEmptyString);
goog.exportSymbol('assertNaN', assertNaN);
goog.exportSymbol('assertNotNaN', assertNotNaN);
goog.exportSymbol('assertObjectEquals', assertObjectEquals);
goog.exportSymbol('assertObjectRoughlyEquals', assertObjectRoughlyEquals);
goog.exportSymbol('assertObjectNotEquals', assertObjectNotEquals);
goog.exportSymbol('assertArrayEquals', assertArrayEquals);
goog.exportSymbol('assertElementsEquals', assertElementsEquals);
goog.exportSymbol('assertElementsRoughlyEqual', assertElementsRoughlyEqual);
goog.exportSymbol('assertSameElements', assertSameElements);
goog.exportSymbol('assertEvaluatesToTrue', assertEvaluatesToTrue);
goog.exportSymbol('assertEvaluatesToFalse', assertEvaluatesToFalse);
goog.exportSymbol('assertHTMLEquals', assertHTMLEquals);
goog.exportSymbol('assertHashEquals', assertHashEquals);
goog.exportSymbol('assertRoughlyEquals', assertRoughlyEquals);
goog.exportSymbol('assertContains', assertContains);
goog.exportSymbol('assertNotContains', assertNotContains);
goog.exportSymbol('assertRegExp', assertRegExp);
