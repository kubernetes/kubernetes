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
 * @fileoverview This file defines base classes used for creating mocks in
 * JavaScript. The API was inspired by EasyMock.
 *
 * The basic API is:
 * <ul>
 *   <li>Create an object to be mocked
 *   <li>Create a mock object, passing in the above object to the constructor
 *   <li>Set expectations by calling methods on the mock object
 *   <li>Call $replay() on the mock object
 *   <li>Pass the mock to code that will make real calls on it
 *   <li>Call $verify() to make sure that expectations were met
 * </ul>
 *
 * For examples, please see the unit tests for LooseMock and StrictMock.
 *
 * Still TODO
 *   implement better (and pluggable) argument matching
 *   Have the exceptions for LooseMock show the number of expected/actual calls
 *   loose and strict mocks share a lot of code - move it to the base class
 *
 */

goog.provide('goog.testing.Mock');
goog.provide('goog.testing.MockExpectation');

goog.require('goog.array');
goog.require('goog.object');
goog.require('goog.testing.JsUnitException');
goog.require('goog.testing.MockInterface');
goog.require('goog.testing.mockmatchers');



/**
 * This is a class that represents an expectation.
 * @param {string} name The name of the method for this expectation.
 * @constructor
 * @final
 */
goog.testing.MockExpectation = function(name) {
  /**
   * The name of the method that is expected to be called.
   * @type {string}
   */
  this.name = name;

  /**
   * An array of error messages for expectations not met.
   * @type {Array}
   */
  this.errorMessages = [];
};


/**
 * The minimum number of times this method should be called.
 * @type {number}
 */
goog.testing.MockExpectation.prototype.minCalls = 1;


/**
  * The maximum number of times this method should be called.
  * @type {number}
  */
goog.testing.MockExpectation.prototype.maxCalls = 1;


/**
 * The value that this method should return.
 * @type {*}
 */
goog.testing.MockExpectation.prototype.returnValue;


/**
 * The value that will be thrown when the method is called
 * @type {*}
 */
goog.testing.MockExpectation.prototype.exceptionToThrow;


/**
 * The arguments that are expected to be passed to this function
 * @type {Array.<*>}
 */
goog.testing.MockExpectation.prototype.argumentList;


/**
 * The number of times this method is called by real code.
 * @type {number}
 */
goog.testing.MockExpectation.prototype.actualCalls = 0;


/**
 * The number of times this method is called during the verification phase.
 * @type {number}
 */
goog.testing.MockExpectation.prototype.verificationCalls = 0;


/**
 * The function which will be executed when this method is called.
 * Method arguments will be passed to this function, and return value
 * of this function will be returned by the method.
 * @type {Function}
 */
goog.testing.MockExpectation.prototype.toDo;


/**
 * Allow expectation failures to include messages.
 * @param {string} message The failure message.
 */
goog.testing.MockExpectation.prototype.addErrorMessage = function(message) {
  this.errorMessages.push(message);
};


/**
 * Get the error messages seen so far.
 * @return {string} Error messages separated by \n.
 */
goog.testing.MockExpectation.prototype.getErrorMessage = function() {
  return this.errorMessages.join('\n');
};


/**
 * Get how many error messages have been seen so far.
 * @return {number} Count of error messages.
 */
goog.testing.MockExpectation.prototype.getErrorMessageCount = function() {
  return this.errorMessages.length;
};



/**
 * The base class for a mock object.
 * @param {Object|Function} objectToMock The object that should be mocked, or
 *    the constructor of an object to mock.
 * @param {boolean=} opt_mockStaticMethods An optional argument denoting that
 *     a mock should be constructed from the static functions of a class.
 * @param {boolean=} opt_createProxy An optional argument denoting that
 *     a proxy for the target mock should be created.
 * @constructor
 * @implements {goog.testing.MockInterface}
 */
goog.testing.Mock = function(objectToMock, opt_mockStaticMethods,
    opt_createProxy) {
  if (!goog.isObject(objectToMock) && !goog.isFunction(objectToMock)) {
    throw new Error('objectToMock must be an object or constructor.');
  }
  if (opt_createProxy && !opt_mockStaticMethods &&
      goog.isFunction(objectToMock)) {
    /**
 * @constructor
 * @final
 */
    var tempCtor = function() {};
    goog.inherits(tempCtor, objectToMock);
    this.$proxy = new tempCtor();
  } else if (opt_createProxy && opt_mockStaticMethods &&
      goog.isFunction(objectToMock)) {
    throw Error('Cannot create a proxy when opt_mockStaticMethods is true');
  } else if (opt_createProxy && !goog.isFunction(objectToMock)) {
    throw Error('Must have a constructor to create a proxy');
  }

  if (goog.isFunction(objectToMock) && !opt_mockStaticMethods) {
    this.$initializeFunctions_(objectToMock.prototype);
  } else {
    this.$initializeFunctions_(objectToMock);
  }
  this.$argumentListVerifiers_ = {};
};


/**
 * Option that may be passed when constructing function, method, and
 * constructor mocks. Indicates that the expected calls should be accepted in
 * any order.
 * @const
 * @type {number}
 */
goog.testing.Mock.LOOSE = 1;


/**
 * Option that may be passed when constructing function, method, and
 * constructor mocks. Indicates that the expected calls should be accepted in
 * the recorded order only.
 * @const
 * @type {number}
 */
goog.testing.Mock.STRICT = 0;


/**
 * This array contains the name of the functions that are part of the base
 * Object prototype.
 * Basically a copy of goog.object.PROTOTYPE_FIELDS_.
 * @const
 * @type {!Array.<string>}
 * @private
 */
goog.testing.Mock.PROTOTYPE_FIELDS_ = [
  'constructor',
  'hasOwnProperty',
  'isPrototypeOf',
  'propertyIsEnumerable',
  'toLocaleString',
  'toString',
  'valueOf'
];


/**
 * A proxy for the mock.  This can be used for dependency injection in lieu of
 * the mock if the test requires a strict instanceof check.
 * @type {Object}
 */
goog.testing.Mock.prototype.$proxy = null;


/**
 * Map of argument name to optional argument list verifier function.
 * @type {Object}
 */
goog.testing.Mock.prototype.$argumentListVerifiers_;


/**
 * Whether or not we are in recording mode.
 * @type {boolean}
 * @private
 */
goog.testing.Mock.prototype.$recording_ = true;


/**
 * The expectation currently being created. All methods that modify the
 * current expectation return the Mock object for easy chaining, so this is
 * where we keep track of the expectation that's currently being modified.
 * @type {goog.testing.MockExpectation}
 * @protected
 */
goog.testing.Mock.prototype.$pendingExpectation;


/**
 * First exception thrown by this mock; used in $verify.
 * @type {Object}
 * @private
 */
goog.testing.Mock.prototype.$threwException_ = null;


/**
 * Initializes the functions on the mock object.
 * @param {Object} objectToMock The object being mocked.
 * @private
 */
goog.testing.Mock.prototype.$initializeFunctions_ = function(objectToMock) {
  // Gets the object properties.
  var enumerableProperties = goog.object.getKeys(objectToMock);

  // The non enumerable properties are added if they override the ones in the
  // Object prototype. This is due to the fact that IE8 does not enumerate any
  // of the prototype Object functions even when overriden and mocking these is
  // sometimes needed.
  for (var i = 0; i < goog.testing.Mock.PROTOTYPE_FIELDS_.length; i++) {
    var prop = goog.testing.Mock.PROTOTYPE_FIELDS_[i];
    // Look at b/6758711 if you're considering adding ALL properties to ALL
    // mocks.
    if (objectToMock[prop] !== Object.prototype[prop]) {
      enumerableProperties.push(prop);
    }
  }

  // Adds the properties to the mock.
  for (var i = 0; i < enumerableProperties.length; i++) {
    var prop = enumerableProperties[i];
    if (typeof objectToMock[prop] == 'function') {
      this[prop] = goog.bind(this.$mockMethod, this, prop);
      if (this.$proxy) {
        this.$proxy[prop] = goog.bind(this.$mockMethod, this, prop);
      }
    }
  }
};


/**
 * Registers a verfifier function to use when verifying method argument lists.
 * @param {string} methodName The name of the method for which the verifierFn
 *     should be used.
 * @param {Function} fn Argument list verifier function.  Should take 2 argument
 *     arrays as arguments, and return true if they are considered equivalent.
 * @return {!goog.testing.Mock} This mock object.
 */
goog.testing.Mock.prototype.$registerArgumentListVerifier = function(methodName,
                                                                     fn) {
  this.$argumentListVerifiers_[methodName] = fn;
  return this;
};


/**
 * The function that replaces all methods on the mock object.
 * @param {string} name The name of the method being mocked.
 * @return {*} In record mode, returns the mock object. In replay mode, returns
 *    whatever the creator of the mock set as the return value.
 */
goog.testing.Mock.prototype.$mockMethod = function(name) {
  try {
    // Shift off the name argument so that args contains the arguments to
    // the mocked method.
    var args = goog.array.slice(arguments, 1);
    if (this.$recording_) {
      this.$pendingExpectation = new goog.testing.MockExpectation(name);
      this.$pendingExpectation.argumentList = args;
      this.$recordExpectation();
      return this;
    } else {
      return this.$recordCall(name, args);
    }
  } catch (ex) {
    this.$recordAndThrow(ex);
  }
};


/**
 * Records the currently pending expectation, intended to be overridden by a
 * subclass.
 * @protected
 */
goog.testing.Mock.prototype.$recordExpectation = function() {};


/**
 * Records an actual method call, intended to be overridden by a
 * subclass. The subclass must find the pending expectation and return the
 * correct value.
 * @param {string} name The name of the method being called.
 * @param {Array} args The arguments to the method.
 * @return {*} The return expected by the mock.
 * @protected
 */
goog.testing.Mock.prototype.$recordCall = function(name, args) {
  return undefined;
};


/**
 * If the expectation expects to throw, this method will throw.
 * @param {goog.testing.MockExpectation} expectation The expectation.
 */
goog.testing.Mock.prototype.$maybeThrow = function(expectation) {
  if (typeof expectation.exceptionToThrow != 'undefined') {
    throw expectation.exceptionToThrow;
  }
};


/**
 * If this expectation defines a function to be called,
 * it will be called and its result will be returned.
 * Otherwise, if the expectation expects to throw, it will throw.
 * Otherwise, this method will return defined value.
 * @param {goog.testing.MockExpectation} expectation The expectation.
 * @param {Array} args The arguments to the method.
 * @return {*} The return value expected by the mock.
 */
goog.testing.Mock.prototype.$do = function(expectation, args) {
  if (typeof expectation.toDo == 'undefined') {
    this.$maybeThrow(expectation);
    return expectation.returnValue;
  } else {
    return expectation.toDo.apply(this, args);
  }
};


/**
 * Specifies a return value for the currently pending expectation.
 * @param {*} val The return value.
 * @return {!goog.testing.Mock} This mock object.
 */
goog.testing.Mock.prototype.$returns = function(val) {
  this.$pendingExpectation.returnValue = val;
  return this;
};


/**
 * Specifies a value for the currently pending expectation to throw.
 * @param {*} val The value to throw.
 * @return {!goog.testing.Mock} This mock object.
 */
goog.testing.Mock.prototype.$throws = function(val) {
  this.$pendingExpectation.exceptionToThrow = val;
  return this;
};


/**
 * Specifies a function to call for currently pending expectation.
 * Note, that using this method overrides declarations made
 * using $returns() and $throws() methods.
 * @param {Function} func The function to call.
 * @return {!goog.testing.Mock} This mock object.
 */
goog.testing.Mock.prototype.$does = function(func) {
  this.$pendingExpectation.toDo = func;
  return this;
};


/**
 * Allows the expectation to be called 0 or 1 times.
 * @return {!goog.testing.Mock} This mock object.
 */
goog.testing.Mock.prototype.$atMostOnce = function() {
  this.$pendingExpectation.minCalls = 0;
  this.$pendingExpectation.maxCalls = 1;
  return this;
};


/**
 * Allows the expectation to be called any number of times, as long as it's
 * called once.
 * @return {!goog.testing.Mock} This mock object.
 */
goog.testing.Mock.prototype.$atLeastOnce = function() {
  this.$pendingExpectation.maxCalls = Infinity;
  return this;
};


/**
 * Allows the expectation to be called exactly once.
 * @return {!goog.testing.Mock} This mock object.
 */
goog.testing.Mock.prototype.$once = function() {
  this.$pendingExpectation.minCalls = 1;
  this.$pendingExpectation.maxCalls = 1;
  return this;
};


/**
 * Disallows the expectation from being called.
 * @return {!goog.testing.Mock} This mock object.
 */
goog.testing.Mock.prototype.$never = function() {
  this.$pendingExpectation.minCalls = 0;
  this.$pendingExpectation.maxCalls = 0;
  return this;
};


/**
 * Allows the expectation to be called any number of times.
 * @return {!goog.testing.Mock} This mock object.
 */
goog.testing.Mock.prototype.$anyTimes = function() {
  this.$pendingExpectation.minCalls = 0;
  this.$pendingExpectation.maxCalls = Infinity;
  return this;
};


/**
 * Specifies the number of times the expectation should be called.
 * @param {number} times The number of times this method will be called.
 * @return {!goog.testing.Mock} This mock object.
 */
goog.testing.Mock.prototype.$times = function(times) {
  this.$pendingExpectation.minCalls = times;
  this.$pendingExpectation.maxCalls = times;
  return this;
};


/**
 * Switches from recording to replay mode.
 * @override
 */
goog.testing.Mock.prototype.$replay = function() {
  this.$recording_ = false;
};


/**
 * Resets the state of this mock object. This clears all pending expectations
 * without verifying, and puts the mock in recording mode.
 * @override
 */
goog.testing.Mock.prototype.$reset = function() {
  this.$recording_ = true;
  this.$threwException_ = null;
  delete this.$pendingExpectation;
};


/**
 * Throws an exception and records that an exception was thrown.
 * @param {string} comment A short comment about the exception.
 * @param {?string=} opt_message A longer message about the exception.
 * @throws {Object} JsUnitException object.
 * @protected
 */
goog.testing.Mock.prototype.$throwException = function(comment, opt_message) {
  this.$recordAndThrow(new goog.testing.JsUnitException(comment, opt_message));
};


/**
 * Throws an exception and records that an exception was thrown.
 * @param {Object} ex Exception.
 * @throws {Object} #ex.
 * @protected
 */
goog.testing.Mock.prototype.$recordAndThrow = function(ex) {
  // If it's an assert exception, record it.
  if (ex['isJsUnitException']) {
    var testRunner = goog.global['G_testRunner'];
    if (testRunner) {
      var logTestFailureFunction = testRunner['logTestFailure'];
      if (logTestFailureFunction) {
        logTestFailureFunction.call(testRunner, ex);
      }
    }

    if (!this.$threwException_) {
      // Only remember first exception thrown.
      this.$threwException_ = ex;
    }
  }
  throw ex;
};


/**
 * Verify that all of the expectations were met. Should be overridden by
 * subclasses.
 * @override
 */
goog.testing.Mock.prototype.$verify = function() {
  if (this.$threwException_) {
    throw this.$threwException_;
  }
};


/**
 * Verifies that a method call matches an expectation.
 * @param {goog.testing.MockExpectation} expectation The expectation to check.
 * @param {string} name The name of the called method.
 * @param {Array.<*>?} args The arguments passed to the mock.
 * @return {boolean} Whether the call matches the expectation.
 */
goog.testing.Mock.prototype.$verifyCall = function(expectation, name, args) {
  if (expectation.name != name) {
    return false;
  }
  var verifierFn =
      this.$argumentListVerifiers_.hasOwnProperty(expectation.name) ?
      this.$argumentListVerifiers_[expectation.name] :
      goog.testing.mockmatchers.flexibleArrayMatcher;

  return verifierFn(expectation.argumentList, args, expectation);
};


/**
 * Render the provided argument array to a string to help
 * clients with debugging tests.
 * @param {Array.<*>?} args The arguments passed to the mock.
 * @return {string} Human-readable string.
 */
goog.testing.Mock.prototype.$argumentsAsString = function(args) {
  var retVal = [];
  for (var i = 0; i < args.length; i++) {
    try {
      retVal.push(goog.typeOf(args[i]));
    } catch (e) {
      retVal.push('[unknown]');
    }
  }
  return '(' + retVal.join(', ') + ')';
};


/**
 * Throw an exception based on an incorrect method call.
 * @param {string} name Name of method called.
 * @param {Array.<*>?} args Arguments passed to the mock.
 * @param {goog.testing.MockExpectation=} opt_expectation Expected next call,
 *     if any.
 */
goog.testing.Mock.prototype.$throwCallException = function(name, args,
                                                           opt_expectation) {
  var errorStringBuffer = [];
  var actualArgsString = this.$argumentsAsString(args);
  var expectedArgsString = opt_expectation ?
      this.$argumentsAsString(opt_expectation.argumentList) : '';

  if (opt_expectation && opt_expectation.name == name) {
    errorStringBuffer.push('Bad arguments to ', name, '().\n',
                           'Actual: ', actualArgsString, '\n',
                           'Expected: ', expectedArgsString, '\n',
                           opt_expectation.getErrorMessage());
  } else {
    errorStringBuffer.push('Unexpected call to ', name,
                           actualArgsString, '.');
    if (opt_expectation) {
      errorStringBuffer.push('\nNext expected call was to ',
                             opt_expectation.name,
                             expectedArgsString);
    }
  }
  this.$throwException(errorStringBuffer.join(''));
};
