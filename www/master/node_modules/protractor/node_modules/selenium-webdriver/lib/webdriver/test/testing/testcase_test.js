// Copyright 2014 Software Freedom Conservancy. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

goog.require('goog.testing.MockControl');
goog.require('goog.testing.PropertyReplacer');
goog.require('goog.testing.mockmatchers');
goog.require('goog.testing.jsunit');
goog.require('goog.testing.recordFunction');
goog.require('webdriver.test.testutil');
goog.require('webdriver.testing.TestCase');
goog.require('webdriver.testing.promise.FlowTester');


// Aliases for readability.
var IGNORE_ARGUMENT = goog.testing.mockmatchers.ignoreArgument,
    IS_ARRAY_ARGUMENT = goog.testing.mockmatchers.isArray,
    STUB_ERROR = webdriver.test.testutil.STUB_ERROR,
    throwStubError = webdriver.test.testutil.throwStubError,
    assertIsStubError = webdriver.test.testutil.assertIsStubError;

var control = new goog.testing.MockControl();
var flowTester, clock, mockTestCase, testStub, mockOnComplete, mockOnError;

function setUp() {
  clock = webdriver.test.testutil.createMockClock();
  flowTester = new webdriver.testing.promise.FlowTester(clock, goog.global);

  // Use one master mock so we can assert execution order.
  mockTestCase = control.createStrictMock({
    setUp: goog.nullFunction,
    testFn: goog.nullFunction,
    tearDown: goog.nullFunction,
    onComplete: goog.nullFunction,
    onError: goog.nullFunction
  }, true);

  mockOnComplete = goog.bind(mockTestCase.onComplete, mockTestCase);
  mockOnError = goog.bind(mockTestCase.onError, mockTestCase);

  testStub = {
    name: 'testStub',
    scope: mockTestCase,
    ref: mockTestCase.testFn
  };

  webdriver.test.testutil.messages = [];
}

function tearDown() {
  flowTester.verifySuccess();
  flowTester.dispose();
  control.$tearDown();
  clock.dispose();
}

function schedule(msg, opt_fn) {
  var fn = opt_fn || goog.nullFunction;
  return webdriver.promise.controlFlow().execute(fn, msg);
}

function runTest() {
  webdriver.testing.TestCase.prototype.runSingleTest_.
      call(mockTestCase, testStub, mockOnError).
      then(mockOnComplete);
  flowTester.run();
  control.$verifyAll();
}

function testExecutesTheBasicTestFlow() {
  mockTestCase.setUp();
  mockTestCase.testFn();
  mockTestCase.tearDown();
  mockOnComplete(IGNORE_ARGUMENT);
  control.$replayAll();

  runTest();
}

function testExecutingAHappyTestWithScheduledActions() {
  mockTestCase.setUp().$does(function() { schedule('a'); });
  mockTestCase.testFn().$does(function() { schedule('b'); });
  mockTestCase.tearDown().$does(function() { schedule('c'); });
  mockOnComplete(IGNORE_ARGUMENT);
  control.$replayAll();

  runTest();
}

function testShouldSkipTestFnIfSetupThrows() {
  mockTestCase.setUp().$does(throwStubError);
  mockOnError(STUB_ERROR);
  mockTestCase.tearDown();
  mockOnComplete(IGNORE_ARGUMENT);
  control.$replayAll();

  runTest();
}

function testShouldSkipTestFnIfSetupActionFails_1() {
  mockTestCase.setUp().$does(function() {
    schedule('an explosion', throwStubError);
  });
  mockOnError(STUB_ERROR);
  mockTestCase.tearDown();
  mockOnComplete(IGNORE_ARGUMENT);
  control.$replayAll();

  runTest();
}

function testShouldSkipTestFnIfSetupActionFails_2() {
  mockTestCase.setUp().$does(function() {
    schedule('an explosion', throwStubError);
  });
  mockOnError(STUB_ERROR);
  mockTestCase.tearDown();
  mockOnComplete(IGNORE_ARGUMENT);
  control.$replayAll();

  runTest();
}

function testShouldSkipTestFnIfNestedSetupActionFails() {
  mockTestCase.setUp().$does(function() {
    schedule('a', goog.nullFunction).then(function() {
      schedule('b', throwStubError);
    });
  });
  mockOnError(STUB_ERROR);
  mockTestCase.tearDown();
  mockOnComplete(IGNORE_ARGUMENT);
  control.$replayAll();

  runTest();
}

function testRunsAllTasksForEachPhaseBeforeTheNextPhase() {
  webdriver.test.testutil.messages = [];
  mockTestCase.setUp().$does(function() { schedule('a'); });
  mockTestCase.testFn().$does(function() { schedule('b'); });
  mockTestCase.tearDown().$does(function() { schedule('c'); });
  mockOnComplete(IGNORE_ARGUMENT);
  control.$replayAll();

  runTest();
}

function testRecordsErrorsFromTestFnBeforeTearDown() {
  mockTestCase.setUp();
  mockTestCase.testFn().$does(throwStubError);
  mockOnError(STUB_ERROR);
  mockTestCase.tearDown();
  mockOnComplete(IGNORE_ARGUMENT);
  control.$replayAll();

  runTest();
}

function testRecordsErrorsFromTearDown() {
  mockTestCase.setUp();
  mockTestCase.testFn();
  mockTestCase.tearDown().$does(throwStubError);
  mockOnError(STUB_ERROR);
  mockOnComplete(IGNORE_ARGUMENT);
  control.$replayAll();

  runTest();
}

function testErrorFromSetUpAndTearDown() {
  mockTestCase.setUp().$does(throwStubError);
  mockOnError(STUB_ERROR);
  mockTestCase.tearDown().$does(throwStubError);
  mockOnError(STUB_ERROR);
  mockOnComplete(IGNORE_ARGUMENT);
  control.$replayAll();

  runTest();
}

function testErrorFromTestFnAndTearDown() {
  var e1 = Error(), e2 = Error();
  mockTestCase.setUp();
  mockTestCase.testFn().$does(function() { throw e1; });
  mockOnError(e1);
  mockTestCase.tearDown().$does(function() { throw e2; });
  mockOnError(e2);
  mockOnComplete(IGNORE_ARGUMENT);
  control.$replayAll();

  runTest();
}
