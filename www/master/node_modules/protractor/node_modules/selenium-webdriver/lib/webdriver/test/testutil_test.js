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

goog.require('goog.testing.jsunit');
goog.require('webdriver.test.testutil');

// Aliases for readability.
var callbackHelper = webdriver.test.testutil.callbackHelper,
    callbackPair = webdriver.test.testutil.callbackPair,
    clock;

function setUp() {
  clock = webdriver.test.testutil.createMockClock();
}

function tearDown() {
  clock.dispose();
}

function testCallbackHelper_functionCalled() {
  var callback = callbackHelper();
  callback();
  assertNotThrows(callback.assertCalled);
  assertThrows(callback.assertNotCalled);
}

function testCallbackHelper_functionCalledMoreThanOnce() {
  var callback = callbackHelper();
  callback();
  callback(123, 'abc');
  assertThrows(callback.assertCalled);
  assertThrows(callback.assertNotCalled);
}

function testCallbackHelper_functionNotCalled() {
  var callback = callbackHelper();
  assertNotThrows(callback.assertNotCalled);
  assertThrows(callback.assertCalled);
}

function testCallbackHelper_wrappedFunctionIsCalled() {
  var count = 0;
  var callback = callbackHelper(function() {
    count += 1;
  });
  callback();
  assertNotThrows(callback.assertCalled);
  assertThrows(callback.assertNotCalled);
  assertEquals(1, count);
}

function testCallbackPair_callbackExpected() {
  var pair = callbackPair();
  assertThrows(pair.assertCallback);
  pair.callback();
  assertNotThrows(pair.assertCallback);
  pair.errback();
  assertThrows(pair.assertCallback);

  pair.reset();
  pair.callback();
  assertNotThrows(pair.assertCallback);
  pair.callback();
  assertThrows('Should expect to be called only once',
      pair.assertCallback);
}

function testCallbackPair_errbackExpected() {
  var pair = callbackPair();
  assertThrows(pair.assertErrback);
  pair.errback();
  assertNotThrows(pair.assertErrback);
  pair.callback();
  assertThrows(pair.assertErrback);
}

function testCallbackPair_eitherExpected() {
  var pair = callbackPair();
  assertThrows(pair.assertEither);
  pair.errback();
  assertNotThrows(pair.assertEither);
  pair.reset();
  pair.callback();
  assertNotThrows(pair.assertEither);
  pair.errback();
  assertNotThrows(pair.assertEither);
}

function testCallbackPair_neitherExpected() {
  var pair = callbackPair();
  assertNotThrows(pair.assertNeither);
  pair.errback();
  assertThrows(pair.assertNeither);
  pair.reset();
  pair.callback();
  assertThrows(pair.assertNeither);
  pair.errback();
  assertThrows(pair.assertNeither);
}

function testZeroBasedTimeoutsRunInNextEventLoop() {
  var count = 0;
  setTimeout(function() {
    count += 1;
    setTimeout(function() { count += 1; }, 0);
    setTimeout(function() { count += 1; }, 0);
  }, 0);
  clock.tick();
  assertEquals(1, count);  // Fails; count == 3
  clock.tick();
  assertEquals(3, count);
}

function testNewZeroBasedTimeoutsRunInNextEventLoopAfterExistingTasks() {
  var events = [];
  setInterval(function() { events.push('a'); }, 1);
  setTimeout(function() { events.push('b'); }, 0);
  clock.tick();
  assertEquals('ab', events.join(''));
}
