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
 * @fileoverview A MockControl holds a set of mocks for a particular test.
 * It consolidates calls to $replay, $verify, and $tearDown, which simplifies
 * the test and helps avoid omissions.
 *
 * You can create and control a mock:
 *   var mockFoo = mockControl.addMock(new MyMock(Foo));
 *
 * MockControl also exposes some convenience functions that create
 * controlled mocks for common mocks: StrictMock, LooseMock,
 * FunctionMock, MethodMock, and GlobalFunctionMock.
 *
 */


goog.provide('goog.testing.MockControl');

goog.require('goog.array');
goog.require('goog.testing');
goog.require('goog.testing.LooseMock');
goog.require('goog.testing.StrictMock');



/**
 * Controls a set of mocks.  Controlled mocks are replayed, verified, and
 * cleaned-up at the same time.
 * @constructor
 */
goog.testing.MockControl = function() {
  /**
   * The list of mocks being controlled.
   * @type {Array.<goog.testing.MockInterface>}
   * @private
   */
  this.mocks_ = [];
};


/**
 * Takes control of this mock.
 * @param {goog.testing.MockInterface} mock Mock to be controlled.
 * @return {goog.testing.MockInterface} The same mock passed in,
 *     for convenience.
 */
goog.testing.MockControl.prototype.addMock = function(mock) {
  this.mocks_.push(mock);
  return mock;
};


/**
 * Calls replay on each controlled mock.
 */
goog.testing.MockControl.prototype.$replayAll = function() {
  goog.array.forEach(this.mocks_, function(m) {
    m.$replay();
  });
};


/**
 * Calls reset on each controlled mock.
 */
goog.testing.MockControl.prototype.$resetAll = function() {
  goog.array.forEach(this.mocks_, function(m) {
    m.$reset();
  });
};


/**
 * Calls verify on each controlled mock.
 */
goog.testing.MockControl.prototype.$verifyAll = function() {
  goog.array.forEach(this.mocks_, function(m) {
    m.$verify();
  });
};


/**
 * Calls tearDown on each controlled mock, if necesssary.
 */
goog.testing.MockControl.prototype.$tearDown = function() {
  goog.array.forEach(this.mocks_, function(m) {
    // $tearDown if defined.
    if (m.$tearDown) {
      m.$tearDown();
    }
    // TODO(user): Somehow determine if verifyAll should have been called
    // but was not.
  });
};


/**
 * Creates a controlled StrictMock.  Passes its arguments through to the
 * StrictMock constructor.
 * @param {Object|Function} objectToMock The object that should be mocked, or
 *    the constructor of an object to mock.
 * @param {boolean=} opt_mockStaticMethods An optional argument denoting that
 *     a mock should be constructed from the static functions of a class.
 * @param {boolean=} opt_createProxy An optional argument denoting that
 *     a proxy for the target mock should be created.
 * @return {!goog.testing.StrictMock} The mock object.
 */
goog.testing.MockControl.prototype.createStrictMock = function(
    objectToMock, opt_mockStaticMethods, opt_createProxy) {
  var m = new goog.testing.StrictMock(objectToMock, opt_mockStaticMethods,
                                      opt_createProxy);
  this.addMock(m);
  return m;
};


/**
 * Creates a controlled LooseMock.  Passes its arguments through to the
 * LooseMock constructor.
 * @param {Object|Function} objectToMock The object that should be mocked, or
 *    the constructor of an object to mock.
 * @param {boolean=} opt_ignoreUnexpectedCalls Whether to ignore unexpected
 *     calls.
 * @param {boolean=} opt_mockStaticMethods An optional argument denoting that
 *     a mock should be constructed from the static functions of a class.
 * @param {boolean=} opt_createProxy An optional argument denoting that
 *     a proxy for the target mock should be created.
 * @return {!goog.testing.LooseMock} The mock object.
 */
goog.testing.MockControl.prototype.createLooseMock = function(
    objectToMock, opt_ignoreUnexpectedCalls,
    opt_mockStaticMethods, opt_createProxy) {
  var m = new goog.testing.LooseMock(objectToMock, opt_ignoreUnexpectedCalls,
                                     opt_mockStaticMethods, opt_createProxy);
  this.addMock(m);
  return m;
};


/**
 * Creates a controlled FunctionMock.  Passes its arguments through to the
 * FunctionMock constructor.
 * @param {string=} opt_functionName The optional name of the function to mock
 *     set to '[anonymous mocked function]' if not passed in.
 * @param {number=} opt_strictness One of goog.testing.Mock.LOOSE or
 *     goog.testing.Mock.STRICT. The default is STRICT.
 * @return {goog.testing.MockInterface} The mocked function.
 */
goog.testing.MockControl.prototype.createFunctionMock = function(
    opt_functionName, opt_strictness) {
  var m = goog.testing.createFunctionMock(opt_functionName, opt_strictness);
  this.addMock(m);
  return m;
};


/**
 * Creates a controlled MethodMock.  Passes its arguments through to the
 * MethodMock constructor.
 * @param {Object} scope The scope of the method to be mocked out.
 * @param {string} functionName The name of the function we're going to mock.
 * @param {number=} opt_strictness One of goog.testing.Mock.LOOSE or
 *     goog.testing.Mock.STRICT. The default is STRICT.
 * @return {!goog.testing.MockInterface} The mocked method.
 */
goog.testing.MockControl.prototype.createMethodMock = function(
    scope, functionName, opt_strictness) {
  var m = goog.testing.createMethodMock(scope, functionName, opt_strictness);
  this.addMock(m);
  return m;
};


/**
 * Creates a controlled MethodMock for a constructor.  Passes its arguments
 * through to the MethodMock constructor. See
 * {@link goog.testing.createConstructorMock} for details.
 * @param {Object} scope The scope of the constructor to be mocked out.
 * @param {string} constructorName The name of the function we're going to mock.
 * @param {number=} opt_strictness One of goog.testing.Mock.LOOSE or
 *     goog.testing.Mock.STRICT. The default is STRICT.
 * @return {!goog.testing.MockInterface} The mocked method.
 */
goog.testing.MockControl.prototype.createConstructorMock = function(
    scope, constructorName, opt_strictness) {
  var m = goog.testing.createConstructorMock(scope, constructorName,
                                             opt_strictness);
  this.addMock(m);
  return m;
};


/**
 * Creates a controlled GlobalFunctionMock.  Passes its arguments through to the
 * GlobalFunctionMock constructor.
 * @param {string} functionName The name of the function we're going to mock.
 * @param {number=} opt_strictness One of goog.testing.Mock.LOOSE or
 *     goog.testing.Mock.STRICT. The default is STRICT.
 * @return {goog.testing.MockInterface} The mocked function.
 */
goog.testing.MockControl.prototype.createGlobalFunctionMock = function(
    functionName, opt_strictness) {
  var m = goog.testing.createGlobalFunctionMock(functionName, opt_strictness);
  this.addMock(m);
  return m;
};
