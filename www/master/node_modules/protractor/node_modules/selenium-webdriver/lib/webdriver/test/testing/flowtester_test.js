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
goog.require('goog.testing.jsunit');
goog.require('webdriver.promise');
goog.require('webdriver.testing.Clock');
goog.require('webdriver.testing.promise.FlowTester');

var FAKE_TIMER = {
  clearInterval: goog.nullFunction,
  clearTimeout: goog.nullFunction,
  setInterval: goog.nullFunction,
  setTimeout: goog.nullFunction
};

var originalDefaultFlow = webdriver.promise.controlFlow();
var originalCreateFlow = webdriver.promise.createFlow;

var control, mockClock;

var flowTester;

function setUp() {
  control = new goog.testing.MockControl();
  mockClock = control.createStrictMock(webdriver.testing.Clock);
  flowTester = new webdriver.testing.promise.FlowTester(
      mockClock, FAKE_TIMER);
}

function tearDown() {
  control.$tearDown();
  flowTester.dispose();
  assertEquals(originalDefaultFlow, webdriver.promise.controlFlow());
  assertEquals(originalCreateFlow, webdriver.promise.createFlow);
}

function testTurnEventLoopAdvancesClockByEventLoopFrequency() {
  mockClock.tick(webdriver.promise.ControlFlow.EVENT_LOOP_FREQUENCY);
  control.$replayAll();

  flowTester.turnEventLoop();
  control.$verifyAll();
}

function captureNewFlow() {
  webdriver.promise.createFlow();
  return goog.array.peek(flowTester.allFlows_).flow;
}

function emitIdle(flow) {
  flow.emit(webdriver.promise.ControlFlow.EventType.IDLE);
}

function emitTask(flow) {
  flow.emit(webdriver.promise.ControlFlow.EventType.SCHEDULE_TASK);
}

function emitError(flow, error) {
  flow.emit(webdriver.promise.ControlFlow.EventType.UNCAUGHT_EXCEPTION,
            error);
}

function testVerifySuccess_aSingleFlow() {
  var verifySuccess = goog.bind(flowTester.verifySuccess, flowTester);
  var flow = webdriver.promise.controlFlow();

  assertNotThrows('Flow has done nothing yet', verifySuccess);

  emitTask(flow);
  assertThrows('Flow is not idle', verifySuccess);

  emitIdle(flow);
  assertNotThrows('Flow went idle', verifySuccess);

  emitError(flow, Error());
  assertThrows('Flow had an error', verifySuccess);

  emitIdle(flow);
  assertThrows('Flow previously had an error', verifySuccess);
}

function testVerifySuccess_multipleFlows() {
  var verifySuccess = goog.bind(flowTester.verifySuccess, flowTester);

  var flow0 = webdriver.promise.controlFlow();
  assertNotThrows('default throw is idle', verifySuccess);

  var flow1 = captureNewFlow();
  assertThrows('New flows start busy', verifySuccess);

  emitIdle(flow1);
  assertNotThrows(verifySuccess);

  assertThrows('Target flow not found', goog.partial(verifySuccess, {}));

  emitError(flow1, Error());
  assertThrows(verifySuccess);

  assertNotThrows(
      'The designated flow is idle and has no errors!',
      goog.partial(verifySuccess, flow0));
}

function testVerifyFailure() {
  var verifyFailure = goog.bind(flowTester.verifyFailure, flowTester);
  var flow0 = webdriver.promise.controlFlow();

  assertThrows('No failures', verifyFailure);

  emitError(flow0, Error());
  assertThrows('Target flow not found', goog.partial(verifyFailure, {}));
  assertNotThrows(verifyFailure);
  assertNotThrows(goog.partial(verifyFailure, flow0));

  emitError(flow0, Error());
  assertThrows('multiple failures', verifyFailure);
}

function testVerifyFailure_multipleFlows() {
  var verifyFailure = goog.bind(flowTester.verifyFailure, flowTester);

  var flow0 = webdriver.promise.controlFlow();
  var flow1 = captureNewFlow();

  emitIdle(flow0);
  emitIdle(flow1);
  assertThrows(verifyFailure);

  emitError(flow0, Error());
  assertNotThrows(verifyFailure);
  assertNotThrows(goog.partial(verifyFailure, flow0));
  assertThrows(goog.partial(verifyFailure, flow1));

  emitError(flow1, Error());
  assertNotThrows(verifyFailure);
  assertNotThrows(goog.partial(verifyFailure, flow0));
  assertNotThrows(goog.partial(verifyFailure, flow1));
}

function testGetFailure() {
  var getFailure = goog.bind(flowTester.getFailure, flowTester);

  var flow0 = webdriver.promise.controlFlow();
  var flow1 = captureNewFlow();

  emitIdle(flow0);
  emitIdle(flow1);
  assertThrows(getFailure);

  var error0 = Error();
  emitError(flow0, error0);
  assertEquals(error0, getFailure());
  assertThrows(goog.partial(getFailure, flow1));

  var error1 = Error();
  emitError(flow1, error1);
  assertThrows(getFailure);
  assertEquals(error0, getFailure(flow0));
  assertEquals(error1, getFailure(flow1));
}

function testAssertStillRunning() {
  var assertStillRunning = goog.bind(
      flowTester.assertStillRunning, flowTester);

  var flow0 = webdriver.promise.controlFlow();
  var flow1 = captureNewFlow();

  assertThrows(assertStillRunning);
  assertThrows(goog.partial(assertStillRunning, flow0));
  assertNotThrows(goog.partial(assertStillRunning, flow1));

  emitIdle(flow1);
  assertThrows(assertStillRunning);
  assertThrows(goog.partial(assertStillRunning, flow0));
  assertThrows(goog.partial(assertStillRunning, flow1));

  emitTask(flow0);
  assertThrows(assertStillRunning);
  assertNotThrows(goog.partial(assertStillRunning, flow0));
  assertThrows(goog.partial(assertStillRunning, flow1));

  emitTask(flow1);
  assertNotThrows(assertStillRunning);
  assertNotThrows(goog.partial(assertStillRunning, flow0));
  assertNotThrows(goog.partial(assertStillRunning, flow1));

  emitError(flow0, Error());
  assertThrows(assertStillRunning);
  assertThrows(goog.partial(assertStillRunning, flow0));
  assertNotThrows(goog.partial(assertStillRunning, flow1));
}
