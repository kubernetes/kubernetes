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

goog.require('goog.array');
goog.require('goog.functions');
goog.require('goog.string');
goog.require('goog.testing.FunctionMock');
goog.require('goog.testing.jsunit');
goog.require('goog.userAgent');
goog.require('webdriver.promise.ControlFlow');
goog.require('webdriver.stacktrace.Snapshot');
goog.require('webdriver.test.testutil');
goog.require('webdriver.testing.promise.FlowTester');

// Aliases for readability.
var STUB_ERROR = webdriver.test.testutil.STUB_ERROR,
    throwStubError = webdriver.test.testutil.throwStubError,
    assertIsStubError = webdriver.test.testutil.assertIsStubError,
    assertingMessages = webdriver.test.testutil.assertingMessages,
    callbackHelper = webdriver.test.testutil.callbackHelper,
    callbackPair = webdriver.test.testutil.callbackPair;

var clock, flow, flowHistory, flowTester;

function shouldRunTests() {
  return !goog.userAgent.IE || goog.userAgent.isVersionOrHigher(10);
}


function setUp() {
  clock = webdriver.test.testutil.createMockClock();
  flowTester = new webdriver.testing.promise.FlowTester(clock, goog.global);
  flow = webdriver.promise.controlFlow();
  webdriver.test.testutil.messages = [];
  flowHistory = [];
}


function tearDown() {
  flowTester.dispose();
  clock.dispose();
}

function schedule(msg, opt_return) {
  return scheduleAction(msg, function() {
    return opt_return;
  });
}

/**
 * @param {string} value The value to push.
 * @param {webdriver.promise.Promise=} opt_taskPromise Promise to return from
 *     the task.
 * @return {!webdriver.promise.Promise} The result.
 */
function schedulePush(value, opt_taskPromise) {
  return scheduleAction(value, function() {
    webdriver.test.testutil.messages.push(value);
    return opt_taskPromise;
  });
}

/**
 * @param {string} msg Debug message.
 * @param {!Function} actionFn The function.
 * @return {!webdriver.promise.Promise} The function result.
 */
function scheduleAction(msg, actionFn) {
  return webdriver.promise.controlFlow().execute(function() {
    flowHistory.push(msg);
    return actionFn();
  }, msg);
}

/**
 * @param {!Function} condition The condition function.
 * @param {number} timeout The timeout.
 * @param {string=} opt_message Optional message.
 * @return {!webdriver.promise.Promise} The wait result.
 */
function scheduleWait(condition, timeout, opt_message) {
  var msg = opt_message || '';
  // It's not possible to hook into when the wait itself is scheduled, so
  // we record each iteration of the wait loop.
  var count = 0;
  return webdriver.promise.controlFlow().wait(function() {
    flowHistory.push((count++) + ': ' + msg);
    return condition();
  }, timeout, msg);
}


/** @see {@link webdriver.testing.promise.FlowTester#turnEventLoop}. */
function turnEventLoop() {
  flowTester.turnEventLoop();
}


function runAndExpectSuccess(opt_callback) {
  flowTester.run();
  flowTester.verifySuccess();
  if (opt_callback) {
    opt_callback();
  }
}


function runAndExpectFailure(opt_errback) {
  flowTester.run();
  flowTester.verifyFailure();
  if (opt_errback) {
    opt_errback(flowTester.getFailure());
  }
}


function assertFlowHistory(var_args) {
  var expected = goog.array.slice(arguments, 0);
  assertArrayEquals(expected, flowHistory);
}


/**
 * @param {string=} opt_description A description of the task for debugging.
 * @return {!webdriver.promise.Task_} The new task.
 */
function createTask(opt_description) {
  return new webdriver.promise.Task_(
      webdriver.promise.controlFlow(),
      goog.nullFunction,
      opt_description || '',
      new webdriver.stacktrace.Snapshot());
}

/**
 * @return {!webdriver.promise.Frame_}
 */
function createFrame() {
  return new webdriver.promise.Frame_(webdriver.promise.controlFlow());
}


function testAddChild_toEmptyFrame() {
  var frame = createFrame();

  var task1 = createTask(),
      task2 = createTask(),
      task3 = createTask();

  frame.addChild(task1);
  frame.addChild(task2);
  frame.addChild(task3);

  assertArrayEquals([task1, task2, task3], frame.children_);
}


function testAddChild_withSubframes() {
  var root = createFrame();

  var task1 = createTask('task1');
  root.addChild(task1);
  assertArrayEquals([task1], root.children_);

  var frame1 = createFrame();
  root.addChild(frame1);
  assertArrayEquals([task1, frame1], root.children_);

  var task2 = createTask('task2'), task3 = createTask('task3');
  root.addChild(task2);
  root.addChild(task3);
  assertArrayEquals([task1, frame1], root.children_);
  assertArrayEquals([task2, task3], frame1.children_);

  frame1.isLocked_ = true;
  var task4 = createTask('task4'), task5 = createTask('task5');
  root.addChild(task4);
  root.addChild(task5);
  assertArrayEquals([task1, frame1, task4, task5], root.children_);
  assertArrayEquals([task2, task3], frame1.children_);

  var frame2 = createFrame(),
      frame3 = createFrame(),
      task6 = createTask('task6'),
      task7 = createTask('task7'),
      task8 = createTask('task8');

  root.addChild(frame2);
  root.addChild(frame3);
  root.addChild(task6);
  frame3.isLocked_ = true;
  root.addChild(task7);
  frame2.isLocked_ = true;
  root.addChild(task8);

  assertArrayEquals([task1, frame1, task4, task5, frame2, task8],
      root.children_);
  assertArrayEquals([task2, task3], frame1.children_);
  assertArrayEquals([frame3, task7], frame2.children_);
  assertArrayEquals([task6], frame3.children_);
}

function testAddChild_insertingFramesIntoAnActiveFrame() {
  var root = createFrame(),
      frame2 = createFrame(),
      frame3 = createFrame(),
      task1 = createTask('task1');

  root.addChild(task1);
  root.isLocked_ = true;
  root.addChild(frame2);
  frame2.isLocked_ = true;
  root.addChild(frame3);
  frame3.isLocked_ = true;

  assertArrayEquals([frame2, frame3, task1], root.children_);
}

function testRemoveChild() {
  var frame1 = createFrame(),
      frame2 = createFrame();

  frame1.addChild(frame2);
  assertArrayEquals([frame2], frame1.children_);
  frame1.removeChild(frame2);
  assertArrayEquals([], frame1.children_);
}


function testResolveFrame() {
  var frame1 = createFrame(),
      frame2 = createFrame(),
      frame3 = createFrame();

  frame2.addChild(frame3);
  frame1.addChild(frame2);
  assertArrayEquals([frame3], frame2.children_);
  assertArrayEquals([frame2], frame1.children_);

  frame1.close = callbackHelper();
  frame2.close = callbackHelper();
  frame3.close = callbackHelper();

  var obj = {
    activeFrame_: frame2,
    commenceShutdown_: callbackHelper(),
    trimHistory_: callbackHelper(),
    history_: []
  };
  webdriver.promise.ControlFlow.prototype.resolveFrame_.call(obj, frame3);
  assertEquals(1, obj.trimHistory_.getCallCount());
  frame3.close.assertCalled('frame 3 not resolved');
  frame2.close.assertNotCalled('frame 2 should not be resolved yet');
  frame1.close.assertNotCalled('frame 1 should not be resolved yet');
  assertNull(frame3.getParent());
  assertArrayEquals([], frame2.children_);
  assertArrayEquals([frame2], frame1.children_);
  assertEquals(frame2, obj.activeFrame_);

  webdriver.promise.ControlFlow.prototype.resolveFrame_.call(obj, frame2);
  assertEquals(2, obj.trimHistory_.getCallCount());
  frame2.close.assertCalled('frame 2 not resolved');
  frame1.close.assertNotCalled('frame 1 should not be resolved yet');
  assertNull(frame2.getParent());
  assertArrayEquals([], frame1.children_);
  assertEquals(frame1, obj.activeFrame_);

  obj.commenceShutdown_.assertNotCalled();
  webdriver.promise.ControlFlow.prototype.resolveFrame_.call(obj, frame1);
  assertEquals(3, obj.trimHistory_.getCallCount());
  frame1.close.assertCalled('frame 1 not resolved');
  obj.commenceShutdown_.assertCalled();
  assertNull(frame1.getParent());
  assertNull(obj.activeFrame_);
}


function testGetNextTask() {
  var root = flow.activeFrame_ = createFrame();

  var frame1 = createFrame(),
      frame2 = createFrame(),
      frame3 = createFrame(),
      task1 = createTask('task1'),
      task2 = createTask('task2'),
      task3 = createTask('task3'),
      task4 = createTask('task4'),
      task5 = createTask('task5'),
      task6 = createTask('task6'),
      task7 = createTask('task7'),
      task8 = createTask('task8');

  flow.commenceShutdown_ = callbackHelper();
  root.close = callbackHelper();
  frame1.close = callbackHelper();
  frame2.close = callbackHelper();
  frame3.close = callbackHelper();

  root.addChild(task1);
  root.addChild(frame1);
  root.addChild(task2);
  root.addChild(task3);
  assertArrayEquals([task1, frame1], root.children_);
  assertArrayEquals([task2, task3], frame1.children_);

  frame1.isLocked_ = true;
  root.addChild(task4);
  root.addChild(task5);
  assertArrayEquals([task1, frame1, task4, task5], root.children_);
  assertArrayEquals([task2, task3], frame1.children_);


  root.addChild(frame2);
  root.addChild(frame3);
  root.addChild(task6);
  frame3.isLocked_ = true;
  root.addChild(task7);
  frame2.isLocked_ = true;
  root.addChild(task8);

  assertArrayEquals([task1, frame1, task4, task5, frame2, task8],
      root.children_);
  assertArrayEquals([task2, task3], frame1.children_);
  assertArrayEquals([frame3, task7], frame2.children_);
  assertArrayEquals([task6], frame3.children_);

  assertEquals(task1, flow.getNextTask_());
  assertEquals(root, flow.activeFrame_);
  root.close.assertNotCalled();
  frame1.close.assertNotCalled();
  frame2.close.assertNotCalled();
  frame3.close.assertNotCalled();

  assertEquals(task2, flow.getNextTask_());
  assertEquals(frame1, flow.activeFrame_);
  root.close.assertNotCalled();
  frame1.close.assertNotCalled();
  frame2.close.assertNotCalled();
  frame3.close.assertNotCalled();

  assertEquals(task3, flow.getNextTask_());
  assertEquals(frame1, flow.activeFrame_);
  root.close.assertNotCalled();
  frame1.close.assertNotCalled();
  frame2.close.assertNotCalled();
  frame3.close.assertNotCalled();

  assertNull(flow.getNextTask_());
  assertNull(frame1.getParent());
  assertEquals(root, flow.activeFrame_);
  root.close.assertNotCalled();
  frame1.close.assertCalled();
  frame2.close.assertNotCalled();
  frame3.close.assertNotCalled();

  assertEquals(task4, flow.getNextTask_());
  assertEquals(root, flow.activeFrame_);
  root.close.assertNotCalled();
  frame2.close.assertNotCalled();
  frame3.close.assertNotCalled();

  assertEquals(task5, flow.getNextTask_());
  assertEquals(root, flow.activeFrame_);
  root.close.assertNotCalled();
  frame2.close.assertNotCalled();
  frame3.close.assertNotCalled();

  assertEquals(task6, flow.getNextTask_());
  assertEquals(frame3, flow.activeFrame_);
  root.close.assertNotCalled();
  frame2.close.assertNotCalled();
  frame3.close.assertNotCalled();

  assertNull(flow.getNextTask_());
  assertNull(frame3.getParent());
  assertEquals(frame2, flow.activeFrame_);
  root.close.assertNotCalled();
  frame2.close.assertNotCalled();
  frame3.close.assertCalled('frame3 should have been resolved');

  assertEquals(task7, flow.getNextTask_());
  assertEquals(frame2, flow.activeFrame_);
  root.close.assertNotCalled();
  frame2.close.assertNotCalled();

  assertNull(flow.getNextTask_());
  assertNull(frame2.getParent());
  assertEquals(root, flow.activeFrame_);
  root.close.assertNotCalled();
  frame2.close.assertCalled('frame2 should have been resolved');

  assertEquals(task8, flow.getNextTask_());
  assertEquals(root, flow.activeFrame_);
  root.close.assertNotCalled();

  flow.commenceShutdown_.assertNotCalled();
  assertNull(flow.getNextTask_());
  assertNull(flow.activeFrame_);
  root.close.assertCalled('Root should have been resolved');
  flow.commenceShutdown_.assertCalled();
}


function testAbortFrame_noActiveFrame() {
  flow.abortFrame_(STUB_ERROR);
  assertIsStubError(flowTester.getFailure());
  assertNull(flow.activeFrame_);
}


function testAbortFrame_activeIsOnlyFrame() {
  // Make the ControlFlow think the flow is not-idle.
  flow.emit(webdriver.promise.ControlFlow.EventType.SCHEDULE_TASK);

  flow.activeFrame_ = createFrame();
  flow.abortFrame_(STUB_ERROR);
  assertNull(flow.activeFrame_);
  flowTester.assertStillRunning();

  clock.tick();
  assertIsStubError(flowTester.getFailure());
}


function testAbortFrame_unhandledAbortionsBubbleUp() {
  var root = flow.activeFrame_ = createFrame(),
      frame1 = createFrame(),
      frame2 = createFrame(),
      frame3 = createFrame(),
      task = createTask();

  var rootHelper = installResolveHelper(root),
      frame1Helper = installResolveHelper(frame1),
      frame2Helper = installResolveHelper(frame2),
      frame3Helper = installResolveHelper(frame3);

  flow.abortNow_ = callbackHelper(assertIsStubError);

  root.addChild(frame1);
  root.addChild(frame2);
  root.addChild(frame3);
  root.addChild(task);

  assertArrayEquals([task], frame3.children_);
  assertArrayEquals([frame3], frame2.children_);
  assertArrayEquals([frame2], frame1.children_);
  assertArrayEquals([frame1], root.children_);

  assertEquals(task, flow.getNextTask_());
  assertEquals(frame3, flow.activeFrame_);
  flow.abortNow_.assertNotCalled();
  rootHelper.assertNeither();
  frame1Helper.assertNeither();
  frame2Helper.assertNeither();
  frame3Helper.assertNeither();

  flow.abortFrame_(STUB_ERROR);
  assertEquals(frame2, flow.activeFrame_);
  flow.abortNow_.assertNotCalled();
  rootHelper.assertNeither();
  frame1Helper.assertNeither();
  frame2Helper.assertNeither();
  frame3Helper.assertErrback();

  clock.tick();
  assertEquals(frame1, flow.activeFrame_);
  flow.abortNow_.assertNotCalled();
  rootHelper.assertNeither();
  frame1Helper.assertNeither();
  frame2Helper.assertErrback();

  clock.tick();
  assertEquals(root, flow.activeFrame_);
  flow.abortNow_.assertNotCalled();
  rootHelper.assertNeither();
  frame1Helper.assertErrback();

  clock.tick();
  assertNull(flow.activeFrame_);
  flow.abortNow_.assertNotCalled();
  rootHelper.assertErrback();

  clock.tick();
  assertNull(flow.activeFrame_);
  flow.abortNow_.assertCalled();

  function installResolveHelper(frame) {
    var abort = goog.bind(frame.abort, frame);
    var close = goog.bind(frame.close, frame);
    var pair = callbackPair(close, function(e) {
      assertIsStubError(e);
      abort(e);
    });
    frame.close = pair.callback;
    frame.abort = pair.errback;
    return pair;
  }
}


function testRunInNewFrame_nothingScheduledInFunction() {
  var root = flow.activeFrame_ = createFrame(),
      task1 = createTask(),
      task2 = createTask();

  root.addChild(task1);
  root.addChild(task2);
  assertArrayEquals([task1, task2], root.children_);

  assertEquals(task1, flow.getNextTask_());
  assertEquals(root, flow.activeFrame_);
  assertArrayEquals([task2], root.children_);

  var pair = callbackPair(assertUndefined);
  flow.runInNewFrame_(goog.nullFunction, pair.callback, pair.errback);
  pair.assertCallback();
  assertEquals(root, flow.activeFrame_);
  assertArrayEquals([task2], root.children_);
}


function testRunInNewFrame_functionThrows() {
  var root = flow.activeFrame_ = createFrame(),
      task1 = createTask(),
      task2 = createTask();

  root.addChild(task1);
  root.addChild(task2);
  assertArrayEquals([task1, task2], root.children_);

  assertEquals(task1, flow.getNextTask_());
  assertEquals(root, flow.activeFrame_);
  assertArrayEquals([task2], root.children_);

  var pair = callbackPair(null, assertIsStubError);
  flow.runInNewFrame_(throwStubError, pair.callback, pair.errback);
  pair.assertErrback();
  assertEquals(root, flow.activeFrame_);
  assertArrayEquals([task2], root.children_);
}


function testRunInNewFrame_functionThrowsAfterSchedulingTasks() {
  var root = flow.activeFrame_ = createFrame(),
      task1 = createTask('task1'),
      task2 = createTask('task2');

  root.addChild(task1);
  root.addChild(task2);
  assertArrayEquals([task1, task2], root.children_);

  assertEquals(task1, flow.getNextTask_());
  assertEquals(root, flow.activeFrame_);
  assertArrayEquals([task2], root.children_);

  var pair = callbackPair(null, assertIsStubError);
  flow.runInNewFrame_(function() {
    flow.execute(goog.nullFunction);
    throw STUB_ERROR;
  }, pair.callback, pair.errback);
  pair.assertErrback();
  assertEquals(root, flow.activeFrame_);
  assertArrayEquals([task2], root.children_);
}


function testRunInNewFrame_whenThereIsNoCurrentActiveFrame_noopFunc() {
  var pair = callbackPair(assertUndefined);
  flow.runInNewFrame_(goog.nullFunction, pair.callback, pair.errback);
  pair.assertCallback();
  assertNull(flow.activeFrame_);
  assertEquals('[]', flow.getSchedule());
}


function testRunInNewFrame_whenThereIsNoCurrentActiveFrame_funcThrows() {
  var pair = callbackPair(null, assertIsStubError);
  flow.runInNewFrame_(throwStubError, pair.callback, pair.errback);
  pair.assertErrback();
  assertNull(flow.activeFrame_);
  assertEquals('[]', flow.getSchedule());
}


function
    testRunInNewFrame_whenThereIsNoCurrentActiveFrame_throwsAfterSchedule() {
  var pair = callbackPair(null, assertIsStubError);
  flow.runInNewFrame_(function() {
    flow.execute('task3', goog.nullFunction);
    throwStubError();
  }, pair.callback, pair.errback);
  pair.assertErrback();
  assertNull(flow.activeFrame_);
  assertEquals('[]', flow.getSchedule());
}


function testRunInNewFrame_returnsPrimitiveFunctionResultImmediately() {
  var pair = callbackPair(goog.partial(assertEquals, 23));
  flow.runInNewFrame_(function() {
    return 23;
  }, pair.callback, pair.errback);
  pair.assertCallback();
}


function testRunInNewFrame_updatesSchedulingFrameForContextOfFunction() {
  var root = flow.activeFrame_ = createFrame();

  var pair = callbackPair();
  flow.runInNewFrame_(function() {
    assertNotNull(flow.activeFrame_);
    assertNotNull(flow.schedulingFrame_);
    assertNotEquals(root, flow.schedulingFrame_);
    assertArrayEquals([flow.schedulingFrame_], root.children_);
    assertEquals(root, flow.schedulingFrame_.getParent());
  }, pair.callback, pair.errback);
  pair.assertCallback();

  assertEquals('Did not restore active frame', root, flow.activeFrame_);
}


function testRunInNewFrame_doesNotReturnUntilScheduledFrameResolved() {
  var root = flow.activeFrame_ = createFrame(),
      task1 = createTask('task1'),
      task2 = createTask('task2');

  root.addChild(task1);
  root.addChild(task2);
  assertArrayEquals([task1, task2], root.children_);

  assertEquals(task1, flow.getNextTask_());
  assertEquals(root, flow.activeFrame_);
  assertArrayEquals([task2], root.children_);

  var pair = callbackPair();
  flow.runInNewFrame_(function() {
    schedule('task3');
  }, pair.callback, pair.errback);

  pair.assertNeither('active frame not resolved yet');
  assertEquals(root, flow.activeFrame_);

  var task = flow.getNextTask_();
  assertEquals('task3', task.getDescription());
  assertEquals(root.children_[0], flow.activeFrame_);
  pair.assertNeither('active frame still not resolved yet');

  assertNull(flow.getNextTask_());
  clock.tick();
  pair.assertCallback();
  assertEquals(root, flow.activeFrame_);
  assertEquals(task2, flow.getNextTask_());
}


function testRunInNewFrame_doesNotReturnUntilScheduledFrameResolved_nested() {
  var root = flow.activeFrame_ = createFrame();

  schedule('task1');
  schedule('task2');
  assertEquals('task1', flow.getNextTask_().getDescription());

  var pair1 = callbackPair(), pair2 = callbackPair();
  flow.runInNewFrame_(function() {
    schedule('task3');
    flow.runInNewFrame_(function() {
      schedule('task4');
    }, pair2.callback, pair2.errback);
  }, pair1.callback, pair1.errback);

  pair1.assertNeither();
  pair2.assertNeither();
  assertEquals('task3', flow.getNextTask_().getDescription());
  assertEquals('task4', flow.getNextTask_().getDescription());
  assertNull(flow.getNextTask_());
  clock.tick();
  pair1.assertNeither();
  pair2.assertCallback();
  assertNull(flow.getNextTask_());
  clock.tick();
  pair1.assertCallback();

  assertEquals(root, flow.activeFrame_);
  assertEquals('task2', flow.getNextTask_().getDescription());
}


function testScheduling_aSimpleFunction() {
  schedule('go');
  runAndExpectSuccess();
  assertFlowHistory('go');
}


function testScheduling_aSimpleSequence() {
  schedule('a');
  schedule('b');
  schedule('c');
  runAndExpectSuccess();
  assertFlowHistory('a', 'b', 'c');
}


function testScheduling_invokesCallbacksWhenTaskIsDone() {
  var callback;
  var d = new webdriver.promise.Deferred();
  schedule('a', d.promise).then(callback = callbackHelper(function(value) {
    assertEquals(123, value);
  }));
  callback.assertNotCalled('Callback should not have been called yet');

  turnEventLoop();
  callback.assertNotCalled('Task has not completed yet!');

  d.fulfill(123);
  callback.assertCalled('Callback should have been called!');
  runAndExpectSuccess();
  assertFlowHistory('a');
}


function testScheduling_blocksUntilPromiseReturnedByTaskIsResolved() {
  var d = new webdriver.promise.Deferred();
  schedule('a', d.promise);
  schedule('b');

  assertFlowHistory();
  turnEventLoop(); assertFlowHistory('a');
  turnEventLoop(); assertFlowHistory('a');  // Task 'a' is still running.
  turnEventLoop(); assertFlowHistory('a');  // Task 'a' is still running.

  d.fulfill(123);
  runAndExpectSuccess();
  assertFlowHistory('a', 'b');
}


function testScheduling_waitsForReturnedPromisesToResolve() {
  var d1 = new webdriver.promise.Deferred();
  var d2 = new webdriver.promise.Deferred();

  var callback;
  schedule('a', d1.promise).then(callback = callbackHelper(function(value) {
    assertEquals('fluffy bunny', value);
  }));

  callback.assertNotCalled('d1 not resolved yet');

  d1.fulfill(d2);
  callback.assertNotCalled('Should not be called yet; blocked on d2');

  d2.fulfill('fluffy bunny');

  runAndExpectSuccess();
  callback.assertCalled('d2 has been resolved');
  assertFlowHistory('a');
}


function testScheduling_executesTasksInAFutureTurnAfterTheyAreScheduled() {
  var count = 0;
  function incr() { count++; }

  scheduleAction('', incr);

  assertEquals(0, count);

  turnEventLoop();
  assertEquals(1, count);

  runAndExpectSuccess();
}


function testScheduling_executesOneTaskPerTurnOfTheEventLoop() {
  var count = 0;
  function incr() { count++; }

  scheduleAction('', incr);
  scheduleAction('', incr);

  assertEquals(0, count);
  turnEventLoop();
  assertEquals(1, count);
  turnEventLoop();
  assertEquals(2, count);

  runAndExpectSuccess();
}


function testScheduling_firstScheduledTaskIsWithinACallback() {
  webdriver.promise.fulfilled().then(function() {
    schedule('a');
    schedule('b');
    schedule('c');
  });
  runAndExpectSuccess();
  assertFlowHistory('a', 'b', 'c');
}


function testFraming_callbacksRunInANewFrame() {
  schedule('a').then(function() {
    schedule('c');
  });
  schedule('b');

  runAndExpectSuccess();
  assertFlowHistory('a', 'c', 'b');
}


function testFraming_lotsOfNesting() {
  schedule('a').then(function() {
    schedule('c').then(function() {
      schedule('e').then(function() {
        schedule('g');
      });
      schedule('f');
    });
    schedule('d');
  });
  schedule('b');

  runAndExpectSuccess();
  assertFlowHistory('a', 'c', 'e', 'g', 'f', 'd', 'b');
}


function testFraming_eachCallbackWaitsForAllScheduledTasksToComplete() {
  schedule('a').
      then(function() {
        schedule('b');
        schedule('c');
      }).
      then(function() {
        schedule('d');
      });
  schedule('e');

  runAndExpectSuccess();
  assertFlowHistory('a', 'b', 'c', 'd', 'e');
}


function testFraming_eachCallbackWaitsForReturnTasksToComplete() {
  schedule('a').
      then(function() {
        schedule('b');
        return schedule('c');
      }).
      then(function() {
        schedule('d');
      });
  schedule('e');

  runAndExpectSuccess();
  assertFlowHistory('a', 'b', 'c', 'd', 'e');
}


function testFraming_promiseCallbacks() {
  webdriver.promise.fulfilled().then(function() {
    schedule('b');
  });
  schedule('a');

  runAndExpectSuccess();
  assertFlowHistory('b', 'a');
}


function testFraming_allCallbacksInAFrameAreScheduledWhenPromiseIsResolved() {
  var a = schedule('a');
  a.then(function() { schedule('b'); });
  schedule('c');
  a.then(function() { schedule('d'); });
  schedule('e');

  runAndExpectSuccess();
  assertFlowHistory('a', 'b', 'd', 'c', 'e');
}


function testFraming_tasksScheduledInInActiveFrameDoNotGetPrecedence() {
  var d = new webdriver.promise.Deferred();

  schedule('a');
  schedule('b');
  d.then(function() { schedule('c'); });

  d.fulfill();
  runAndExpectSuccess();
  assertFlowHistory('a', 'b', 'c');
}


function testFraming_tasksScheduledInAFrameGetPrecedence_1() {
  var a = schedule('a');
  schedule('b').then(function() {
    a.then(function() {
      schedule('c');
      schedule('d');
    });
    var e = schedule('e');
    a.then(function() {
      // When this function runs, |e| will not be resolved yet, so |f| and
      // |h| will be resolved first.  After |e| is resolved, |g| will be
      // scheduled in a new frame, resulting in: [j][f, h, i][g], so |g| is
      // expected to execute first.
      schedule('f');
      e.then(function() {
        schedule('g');
      });
      schedule('h');
    });
    schedule('i');
  });
  schedule('j');

  runAndExpectSuccess();
  assertFlowHistory('a', 'b', 'c', 'd', 'e', 'g', 'f', 'h', 'i', 'j');
}


function testErrorHandling_thrownErrorsArePassedToTaskErrback() {
  var callbacks = callbackPair(null, assertIsStubError);
  scheduleAction('function that throws', throwStubError).
      then(callbacks.callback, callbacks.errback);
  runAndExpectSuccess(callbacks.assertErrback);
}


function testErrorHandling_thrownErrorsPropagateThroughPromiseChain() {
  var callbacks = callbackPair(null, assertIsStubError);
  scheduleAction('function that throws', throwStubError).
      then(callbacks.callback).
      then(null, callbacks.errback);
  runAndExpectSuccess(callbacks.assertErrback);
}


function testErrorHandling_catchesErrorsFromFailedTasksInAFrame() {
  var errback;

  schedule('a').
      then(function() {
        schedule('b');
        scheduleAction('function that throws', throwStubError);
      }).
      then(null, errback = callbackHelper(assertIsStubError));

  runAndExpectSuccess();
  errback.assertCalled();
}


function testErrorHandling_abortsIfOnlyTaskThrowsAnError() {
  scheduleAction('function that throws', throwStubError);
  runAndExpectFailure(assertIsStubError);
}


function testErrorHandling_abortsIfOnlyTaskReturnsAnUnhandledRejection() {
  var rejected = webdriver.promise.rejected(STUB_ERROR);
  scheduleAction('function that throws', function() { return rejected; });
  runAndExpectFailure(assertIsStubError);
}


function testErrorHandling_abortsIfThereIsAnUnhandledRejection() {
  webdriver.promise.rejected(STUB_ERROR);
  schedule('this should not run');
  runAndExpectFailure(assertIsStubError);
  assertFlowHistory();
}


function testErrorHandling_abortsSequenceIfATaskFails() {
  schedule('a');
  schedule('b');
  scheduleAction('c', throwStubError);
  schedule('d');  // Should never execute.

  runAndExpectFailure(assertIsStubError);
  assertFlowHistory('a', 'b', 'c');
}


function testErrorHandling_abortsFromUnhandledFramedTaskFailures_1() {
  schedule('outer task').then(function() {
    scheduleAction('inner task', throwStubError);
  });
  schedule('this should not run');
  runAndExpectFailure(assertIsStubError);
  assertFlowHistory('outer task', 'inner task');
}


function testErrorHandling_abortsFromUnhandledFramedTaskFailures_2() {
  schedule('a').then(function() {
    schedule('b').then(function() {
      scheduleAction('c', throwStubError);
      // This should not execute.
      schedule('d');
    });
  });

  runAndExpectFailure(assertIsStubError);
  assertFlowHistory('a', 'b', 'c');
}


function testErrorHandling_abortsWhenErrorBubblesUpFromFullyResolvingAnObject() {
  var obj = {'foo': webdriver.promise.rejected(STUB_ERROR)};
  scheduleAction('', function() {
    return webdriver.promise.fullyResolved(obj).
        then(function() {
          // Should never get here; STUB_ERROR should abort the flow above.
          return webdriver.promise.rejected('rejected 2');
        });
  });
  runAndExpectFailure(assertIsStubError);
}


function testErrorHandling_abortsWhenErrorBubblesUpFromFullyResolvingAnObject_withCallback() {
  var obj = {'foo': webdriver.promise.rejected(STUB_ERROR)};
  var callback;
  scheduleAction('', function() {
    return webdriver.promise.fullyResolved(obj).
        then(function() {
          // Should never get here; STUB_ERROR should abort the flow above.
          return webdriver.promise.rejected('rejected 2');
        });
  }).then(callback = callbackHelper());

  callback.assertNotCalled();
  runAndExpectFailure(assertIsStubError);
}


function testErrorHandling_canCatchErrorsFromNestedTasks() {
  var errback;
  schedule('a').
      then(function() {
        return scheduleAction('b', throwStubError);
      }).
      thenCatch(errback = callbackHelper(assertIsStubError));
  runAndExpectSuccess();
  errback.assertCalled();
}


function testErrorHandling_nestedCommandFailuresCanBeCaughtAndSuppressed() {
  var errback;
  schedule('a').then(function() {
    return schedule('b').then(function() {
      return schedule('c').then(function() {
        throw STUB_ERROR;
      });
    });
  }).thenCatch(errback = callbackHelper(assertIsStubError));
  schedule('d');
  runAndExpectSuccess();
  assertFlowHistory('a', 'b', 'c', 'd');
  errback.assertCalled();
}


function testErrorHandling_aTaskWithAnUnhandledPromiseRejection() {
  schedule('a');
  scheduleAction('sub-tasks', function() {
    webdriver.promise.rejected(STUB_ERROR);
  });
  schedule('should never run');

  runAndExpectFailure(assertIsStubError);
  assertFlowHistory('a', 'sub-tasks');
}

function testErrorHandling_aTaskThatReutrnsARejectedPromise() {
  schedule('a');
  scheduleAction('sub-tasks', function() {
    return webdriver.promise.rejected(STUB_ERROR);
  });
  schedule('should never run');

  runAndExpectFailure(assertIsStubError);
  assertFlowHistory('a', 'sub-tasks')
}


function testErrorHandling_discardsSubtasksIfTaskThrows() {
  var pair = callbackPair(null, assertIsStubError);
  scheduleAction('a', function() {
    schedule('b');
    schedule('c');
    throwStubError();
  }).then(pair.callback, pair.errback);
  schedule('d');

  runAndExpectSuccess();
  pair.assertErrback();
  assertFlowHistory('a', 'd');
}


function testErrorHandling_discardsRemainingSubtasksIfASubtaskFails() {
  var pair = callbackPair(null, assertIsStubError);
  scheduleAction('a', function() {
    schedule('b');
    scheduleAction('c', throwStubError);
    schedule('d');
  }).then(pair.callback, pair.errback);
  schedule('e');

  runAndExpectSuccess();
  pair.assertErrback();
  assertFlowHistory('a', 'b', 'c', 'e');
}


function testTryFinally_happyPath() {
  /* Model:
     try {
       doFoo();
       doBar();
     } finally {
       doBaz();
     }
   */
  schedulePush('foo').
      then(goog.partial(schedulePush, 'bar')).
      thenFinally(goog.partial(schedulePush, 'baz'));
  runAndExpectSuccess(assertingMessages('foo', 'bar', 'baz'));
  assertFlowHistory('foo', 'bar', 'baz');
}


function testTryFinally_firstTryFails() {
  /* Model:
     try {
       doFoo();
       doBar();
     } finally {
       doBaz();
     }
   */

  scheduleAction('doFoo and throw', function() {
    webdriver.test.testutil.messages.push('foo');
    throw STUB_ERROR;
  }).then(goog.partial(schedulePush, 'bar')).
      thenFinally(goog.partial(schedulePush, 'baz'));
  runAndExpectFailure(function(e) {
    assertIsStubError(e);
    webdriver.test.testutil.assertMessages('foo', 'baz');
  });
}


function testTryFinally_secondTryFails() {
  /* Model:
     try {
       doFoo();
       doBar();
     } finally {
       doBaz();
     }
   */

  schedulePush('foo').
      then(function() {
        return scheduleAction('doBar and throw', function() {
          webdriver.test.testutil.messages.push('bar');
          throw STUB_ERROR;
        });
      }).
      thenFinally(function() {
        return schedulePush('baz');
      });
  runAndExpectFailure(function(e) {
    assertIsStubError(e);
    webdriver.test.testutil.assertMessages('foo', 'bar' , 'baz');
  });
}


function testDelayedNesting_1() {
  var a = schedule('a');
  schedule('b').then(function() {
    a.then(function() { schedule('c'); });
    schedule('d');
  });
  schedule('e');

  runAndExpectSuccess();
  assertFlowHistory('a', 'b', 'c', 'd', 'e');
}


function testDelayedNesting_2() {
  var a = schedule('a');
  schedule('b').then(function() {
    a.then(function() { schedule('c'); });
    schedule('d');
    a.then(function() { schedule('e'); });
  });
  schedule('f');

  runAndExpectSuccess();
  assertFlowHistory('a', 'b', 'c', 'd', 'e', 'f');
}


function testDelayedNesting_3() {
  var a = schedule('a');
  schedule('b').then(function() {
    a.then(function() { schedule('c'); });
    a.then(function() { schedule('d'); });
  });
  schedule('e');

  runAndExpectSuccess();
  assertFlowHistory('a', 'b', 'c', 'd', 'e');
}


function testDelayedNesting_4() {
  var a = schedule('a');
  schedule('b').then(function() {
    a.then(function() { schedule('c'); }).then(function() {
      schedule('d');
    });
    a.then(function() { schedule('e'); });
  });
  schedule('f');

  runAndExpectSuccess();
  assertFlowHistory('a', 'b', 'c', 'd', 'e', 'f');
}


function testDelayedNesting_5() {
  var a = schedule('a');
  schedule('b').then(function() {
    var c;
    a.then(function() { c = schedule('c'); }).then(function() {
      schedule('d');
      a.then(function() { schedule('e'); });
      c.then(function() { schedule('f'); });
      schedule('g');
    });
    a.then(function() { schedule('h'); });
  });
  schedule('i');

  runAndExpectSuccess();
  assertFlowHistory('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i');
}


function testCancelsTerminationEventIfNewCommandIsScheduled() {
  schedule('a');
  turnEventLoop();
  assertFlowHistory('a');
  flowTester.assertStillRunning();
  turnEventLoop();
  schedule('b');

  runAndExpectSuccess();
  assertFlowHistory('a', 'b');
}

function testWaiting_onAConditionThatIsAlwaysTrue() {
  scheduleWait(function() { return true;}, 0, 'waiting on true');
  runAndExpectSuccess();
  assertFlowHistory('0: waiting on true');
}


function testWaiting_aSimpleCountingCondition() {
  var count = 0;
  scheduleWait(function() {
    return ++count == 3;
  }, 200, 'counting to 3');

  turnEventLoop();  // Start the flow; triggers first condition poll.
  assertEquals(1, count);
  clock.tick(100);  // Poll 2 more times.
  clock.tick(100);
  assertEquals(3, count);

  runAndExpectSuccess();
}


function testWaiting_aConditionThatReturnsAPromise() {
  var d = new webdriver.promise.Deferred();

  scheduleWait(function() {
    return d.promise;
  }, 0, 'waiting for promise');

  turnEventLoop();
  flowTester.assertStillRunning();

  // Should be able to turn the event loop a few times since we're blocked
  // on our wait condition.
  turnEventLoop();
  turnEventLoop();

  d.fulfill(123);
  runAndExpectSuccess();
}


function testWaiting_aConditionThatReturnsAPromise_2() {
  var count = 0;
  scheduleWait(function() {
    return webdriver.promise.fulfilled(++count == 3);
  }, 200, 'waiting for promise');

  turnEventLoop();  // Start the flow; triggers first condition poll.
  clock.tick(100);  // Poll 2 more times.
  clock.tick(100);
  assertEquals(3, count);

  runAndExpectSuccess();
}


function testWaiting_aConditionThatReturnsATaskResult() {
  var count = 0;
  scheduleWait(function() {
    return scheduleAction('increment count', function() {
      return ++count == 3;
    });
  }, 200, 'counting to 3');
  schedule('post wait');

  turnEventLoop();
  assertEquals(0, count);
  assertFlowHistory('0: counting to 3');

  turnEventLoop();  // Runs scheduled task.
  turnEventLoop();
  assertFlowHistory(
      '0: counting to 3', 'increment count');
  assertEquals(1, count);

  clock.tick(100);  // Advance clock for next polling pass.
  assertEquals(1, count);
  turnEventLoop();
  clock.tick();
  assertEquals(2, count);
  turnEventLoop();
  assertFlowHistory(
      '0: counting to 3', 'increment count',
      '1: counting to 3', 'increment count');

  clock.tick(100);  // Advance clock for next polling pass.
  assertEquals(2, count);
  turnEventLoop();
  clock.tick();
  assertEquals(3, count);
  turnEventLoop();
  assertFlowHistory(
      '0: counting to 3', 'increment count',
      '1: counting to 3', 'increment count',
      '2: counting to 3', 'increment count');

  runAndExpectSuccess();
  assertEquals(3, count);
  assertFlowHistory(
      '0: counting to 3', 'increment count',
      '1: counting to 3', 'increment count',
      '2: counting to 3', 'increment count',
      'post wait');
}


function testWaiting_conditionContainsASubtask() {
  var count = 0;
  scheduleWait(function() {
    schedule('sub task');
    return ++count == 3;
  }, 200, 'counting to 3');
  schedule('post wait');

  runAndExpectSuccess();
  assertEquals(3, count);
  assertFlowHistory(
      '0: counting to 3', 'sub task',
      '1: counting to 3', 'sub task',
      '2: counting to 3', 'sub task',
      'post wait');
}


function testWaiting_cancelsWaitIfScheduledTaskFails() {
  var pair = callbackPair(null, assertIsStubError);
  scheduleWait(function() {
    scheduleAction('boom', throwStubError);
    schedule('this should not run');
    return true;
  }, 200, 'waiting to go boom').then(pair.callback, pair.errback);
  schedule('post wait');

  runAndExpectSuccess();
  assertFlowHistory(
      '0: waiting to go boom', 'boom',
      'post wait');
}


function testWaiting_failsIfConditionThrows() {
  var callbacks = callbackPair(null, assertIsStubError);
  scheduleWait(throwStubError, 0, 'goes boom').
      then(callbacks.callback, callbacks.errback);
  schedule('post wait');
  runAndExpectSuccess();
  assertFlowHistory('0: goes boom', 'post wait');
  callbacks.assertErrback();
}


function testWaiting_failsIfConditionReturnsARejectedPromise() {
  var callbacks = callbackPair(null, assertIsStubError);
  scheduleWait(function() {
    return webdriver.promise.rejected(STUB_ERROR);
  }, 0, 'goes boom').then(callbacks.callback, callbacks.errback);
  schedule('post wait');
  runAndExpectSuccess();
  assertFlowHistory('0: goes boom', 'post wait');
  callbacks.assertErrback();
}


function testWaiting_failsIfConditionHasUnhandledRejection() {
  var callbacks = callbackPair(null, assertIsStubError);
  scheduleWait(function() {
    webdriver.promise.controlFlow().execute(throwStubError);
  }, 0, 'goes boom').then(callbacks.callback, callbacks.errback);
  schedule('post wait');
  runAndExpectSuccess();
  assertFlowHistory('0: goes boom', 'post wait');
  callbacks.assertErrback();
}


function testWaiting_failsIfConditionHasAFailedSubtask() {
  var callbacks = callbackPair(null, assertIsStubError);
  var count = 0;
  scheduleWait(function() {
    scheduleAction('maybe throw', function() {
      if (++count == 2) {
        throw STUB_ERROR;
      }
    });
  }, 200, 'waiting').then(callbacks.callback, callbacks.errback);
  schedule('post wait');

  turnEventLoop();
  assertEquals(0, count);

  turnEventLoop();  // Runs scheduled task.
  assertEquals(1, count);

  clock.tick(100);  // Advance clock for next polling pass.
  assertEquals(1, count);

  runAndExpectSuccess();
  assertEquals(2, count);
  assertFlowHistory(
      '0: waiting', 'maybe throw',
      '1: waiting', 'maybe throw',
      'post wait');
}


function testWaiting_pollingLoopWaitsForAllScheduledTasksInCondition() {
  var count = 0;
  scheduleWait(function() {
    scheduleAction('increment count', function() { ++count; });
    return count >= 3;
  }, 350, 'counting to 3');
  schedule('post wait');

  turnEventLoop();
  assertEquals(0, count);

  turnEventLoop();  // Runs scheduled task.
  turnEventLoop();
  assertEquals(1, count);

  clock.tick(100);  // Advance clock for next polling pass.
  assertEquals(1, count);
  turnEventLoop();
  clock.tick();
  assertEquals(2, count);

  clock.tick(100);  // Advance clock for next polling pass.
  assertEquals(2, count);

  runAndExpectSuccess();
  assertEquals(4, count);
  assertFlowHistory(
      '0: counting to 3', 'increment count',
      '1: counting to 3', 'increment count',
      '2: counting to 3', 'increment count',
      '3: counting to 3', 'increment count',
      'post wait');
}


function testWaiting_blocksNextTaskOnWait() {
  var count = 0;
  scheduleWait(function() {
    return ++count == 3;
  }, 200, 'counting to 3');
  schedule('post wait');

  turnEventLoop();  // Start the flow; triggers first condition poll.
  assertFlowHistory('0: counting to 3');
  assertEquals(1, count);
  clock.tick(100);  // Poll 2 more times.
  assertFlowHistory(
      '0: counting to 3',
      '1: counting to 3');
  clock.tick(100);
  assertFlowHistory(
      '0: counting to 3',
      '1: counting to 3',
      '2: counting to 3');
  assertEquals(3, count);

  runAndExpectSuccess();
  assertFlowHistory(
      '0: counting to 3',
      '1: counting to 3',
      '2: counting to 3',
      'post wait');
}


function testWaiting_timesOut_zeroTimeout() {
  scheduleWait(function() { return false; }, 0, 'always false');
  runAndExpectFailure(goog.nullFunction);
}

function testWaiting_timesOut_nonZeroTimeout() {
  var count = 0;
  scheduleWait(function() {
    return ++count == 3;
  }, 100, 'counting to 3');

  turnEventLoop();  // Start the flow; triggers first condition poll.
  clock.tick(100);  // Poll 2 more times.
  assertEquals(2, count);

  runAndExpectFailure(function() {
    assertFlowHistory('0: counting to 3', '1: counting to 3');
    assertEquals(2, count);
  });
}


function testWaiting_shouldFailIfConditionReturnsARejectedPromise() {
  var count = 0;
  scheduleWait(function() {
    return webdriver.promise.rejected(STUB_ERROR);
  }, 100, 'counting to 3');

  runAndExpectFailure(assertIsStubError);
}


function testWaiting_callbacks() {
  var pair = callbackPair();

  scheduleWait(function() { return true;}, 0, 'waiting on true').
      then(pair.callback, pair.errback);
  pair.assertNeither('Wait not expected to be done yet');
  turnEventLoop();
  pair.assertCallback('Wait callback not called!');
  runAndExpectSuccess();
}


function testWaiting_errbacks() {
  scheduleWait(function() { return false; }, 0, 'always false');

  runAndExpectFailure();
}


function testWaiting_scheduleWithIntermittentWaits() {
  schedule('a');
  scheduleWait(function() { return true; }, 0, 'wait 1');
  schedule('b');
  scheduleWait(function() { return true; }, 0, 'wait 2');
  schedule('c');
  scheduleWait(function() { return true; }, 0, 'wait 3');

  runAndExpectSuccess();
  assertFlowHistory('a', '0: wait 1', 'b', '0: wait 2', 'c', '0: wait 3');
}


function testWaiting_scheduleWithIntermittentAndNestedWaits() {
  schedule('a');
  scheduleWait(function() { return true; }, 0, 'wait 1').
      then(function() {
        schedule('d');
        scheduleWait(function() { return true; }, 0, 'wait 2');
        schedule('e');
      });
  schedule('b');
  scheduleWait(function() { return true; }, 0, 'wait 3');
  schedule('c');
  scheduleWait(function() { return true; }, 0, 'wait 4');

  runAndExpectSuccess();
  assertFlowHistory(
      'a', '0: wait 1', 'd', '0: wait 2', 'e', 'b', '0: wait 3', 'c',
      '0: wait 4');
}


function testSubtasks() {
  schedule('a');
  scheduleAction('sub-tasks', function() {
    schedule('c');
    schedule('d');
  });
  schedule('b');

  runAndExpectSuccess();
  assertFlowHistory('a', 'sub-tasks', 'c', 'd', 'b');
}


function testSubtasks_nesting() {
  schedule('a');
  scheduleAction('sub-tasks', function() {
    schedule('b');
    scheduleAction('sub-sub-tasks', function() {
      schedule('c');
      schedule('d');
    });
    schedule('e');
  });
  schedule('f');

  runAndExpectSuccess();
  assertFlowHistory(
      'a', 'sub-tasks', 'b', 'sub-sub-tasks', 'c', 'd', 'e', 'f');
}


function testSubtasks_taskReturnsSubTaskResult_1() {
  schedule('a');
  scheduleAction('sub-tasks', function() {
    return schedule('c');
  });
  schedule('b');

  runAndExpectSuccess();
  assertFlowHistory('a', 'sub-tasks', 'c', 'b');
}


function testSubtasks_taskReturnsSubTaskResult_2() {
  var callback;
  schedule('a');
  schedule('sub-tasks', webdriver.promise.fulfilled(123)).
      then(callback = callbackHelper(function(value) {
        assertEquals(123, value);
      }));
  schedule('b');

  runAndExpectSuccess();
  assertFlowHistory('a', 'sub-tasks','b');
  callback.assertCalled();
}


function testSubtasks_subTaskFails_1() {
  schedule('a');
  scheduleAction('sub-tasks', function() {
    scheduleAction('sub-task that fails', throwStubError);
  });
  schedule('should never execute');

  runAndExpectFailure(assertIsStubError);
  assertFlowHistory('a', 'sub-tasks', 'sub-task that fails');
}


function testSubtasks_subTaskFails_2() {
  schedule('a');
  scheduleAction('sub-tasks', function() {
    return webdriver.promise.rejected(STUB_ERROR);
  });
  schedule('should never execute');

  runAndExpectFailure(assertIsStubError);
  assertFlowHistory('a', 'sub-tasks');
}


function testSubtasks_subTaskFails_3() {
  var callbacks = callbackPair(null, assertIsStubError);

  schedule('a');
  scheduleAction('sub-tasks', function() {
    return webdriver.promise.rejected(STUB_ERROR);
  }).then(callbacks.callback, callbacks.errback);
  schedule('b');

  runAndExpectSuccess();
  assertFlowHistory('a', 'sub-tasks', 'b');
  callbacks.assertErrback();
}


function testEventLoopWaitsOnPendingPromiseRejections_oneRejection() {
  var d = new webdriver.promise.Deferred;
  scheduleAction('one', function() {
    return d.promise;
  });
  scheduleAction('two', goog.nullFunction);

  turn();
  assertFlowHistory('one');
  turn(-1);
  d.reject(STUB_ERROR);
  clock.tick(1);
  assertFlowHistory('one');
  runAndExpectFailure(assertIsStubError);
  assertFlowHistory('one');

  function turn(opt_minusN) {
    var n = webdriver.promise.ControlFlow.EVENT_LOOP_FREQUENCY;
    if (opt_minusN) n -= Math.abs(opt_minusN);
    clock.tick(n);
  }
}


function testEventLoopWaitsOnPendingPromiseRejections_multipleRejections() {
  var once = Error('once');
  var twice = Error('twice');
  var onError = new goog.testing.FunctionMock('onError',
      goog.testing.Mock.LOOSE);
  onError(once);
  onError(twice);
  onError.$replay();

  flow.on(
      webdriver.promise.ControlFlow.EventType.UNCAUGHT_EXCEPTION, onError);

  scheduleAction('one', goog.nullFunction);
  scheduleAction('two', goog.nullFunction);

  turn();
  assertFlowHistory('one');
  turn(-1);
  webdriver.promise.rejected(once);
  webdriver.promise.rejected(twice);
  clock.tick(1);
  assertFlowHistory('one');
  turn();
  onError.$verify();

  function turn(opt_minusN) {
    var n = webdriver.promise.ControlFlow.EVENT_LOOP_FREQUENCY;
    if (opt_minusN) n -= Math.abs(opt_minusN);
    clock.tick(n);
  }
}

function testCancelsPromiseReturnedByCallbackIfFrameFails_promiseCallback() {
  var isCancellationError = function(e) {
    assertEquals('CancellationError: Error: ouch', e.toString());
  };

  var chainPair = callbackPair(null, isCancellationError);
  var deferredPair = callbackPair(null, isCancellationError);

  var d = new webdriver.promise.Deferred();
  d.then(deferredPair.callback, deferredPair.errback);

  webdriver.promise.fulfilled().
      then(function() {
        scheduleAction('boom', throwStubError);
        schedule('this should not run');
        return d.promise;
      }).
      then(chainPair.callback, chainPair.errback);

  runAndExpectSuccess();
  assertFlowHistory('boom');
  chainPair.assertErrback('chain errback not invoked');
  deferredPair.assertErrback('deferred errback not invoked');
}

function testCancelsPromiseReturnedByCallbackIfFrameFails_taskCallback() {
  var isCancellationError = function(e) {
    assertEquals('CancellationError: Error: ouch', e.toString());
  };

  var chainPair = callbackPair(null, isCancellationError);
  var deferredPair = callbackPair(null, isCancellationError);

  var d = new webdriver.promise.Deferred();
  d.then(deferredPair.callback, deferredPair.errback);

  schedule('a').
      then(function() {
        scheduleAction('boom', throwStubError);
        schedule('this should not run');
        return d.promise;
      }).
      then(chainPair.callback, chainPair.errback);

  runAndExpectSuccess();
  assertFlowHistory('a', 'boom');
  chainPair.assertErrback('chain errback not invoked');
  deferredPair.assertErrback('deferred errback not invoked');
}

function testMaintainsOrderInCallbacksWhenATaskReturnsAPromise() {
  schedule('__start__', webdriver.promise.fulfilled()).
      then(function() {
        webdriver.test.testutil.messages.push('a');
        schedulePush('b');
        webdriver.test.testutil.messages.push('c');
      }).
      then(function() {
        webdriver.test.testutil.messages.push('d');
      });
  schedulePush('e');

  runAndExpectSuccess();
  assertFlowHistory('__start__', 'b', 'e');
  webdriver.test.testutil.assertMessages('a', 'c', 'b', 'd', 'e');
}

function assertFrame(description, frame) {
  var regexp = new RegExp('^' + description + '(\\n    at .*)*$');
  assertTrue(
      'Frame did not match expected regex:' +
          '\n expected: ' + regexp +
          '\n was: ' + frame,
     regexp.test(frame));
}

function testHistory_removesLastTaskEachTimeANewTaskIsStarted() {
  schedule('one').then(function() {
    var flowHistory = webdriver.promise.controlFlow().getHistory();
    assertEquals(1, flowHistory.length);
    assertFrame('one', flowHistory[0]);
  });
  schedule('two').then(function() {
    var flowHistory = webdriver.promise.controlFlow().getHistory();
    assertEquals(1, flowHistory.length);
    assertFrame('two', flowHistory[0]);
  });
  schedule('three').then(function() {
    var flowHistory = webdriver.promise.controlFlow().getHistory();
    assertEquals(1, flowHistory.length);
    assertFrame('three', flowHistory[0]);
  });
  runAndExpectSuccess();
  assertEquals(0, webdriver.promise.controlFlow().getHistory().length);
}

function testHistory_clearsSubtaskHistoryWhenParentTaskCompletes() {
  scheduleAction('one', function() {
    schedule('two').then(function() {
      var flowHistory = webdriver.promise.controlFlow().getHistory();
      assertEquals(2, flowHistory.length);
      assertFrame('two', flowHistory[0]);
      assertFrame('one', flowHistory[1]);
    });
  }).then(function() {
    var flowHistory = webdriver.promise.controlFlow().getHistory();
    assertEquals(1, flowHistory.length);
    assertFrame('one', flowHistory[0]);
  });
  runAndExpectSuccess();
  assertFlowHistory('one', 'two');
  assertEquals(0, webdriver.promise.controlFlow().getHistory().length);
}

function testHistory_preservesHistoryWhenChildTaskFails() {
  scheduleAction('one', function() {
    scheduleAction('two', function() {
      scheduleAction('three', throwStubError);
    });
  }).then(fail, function() {
    var flowHistory = webdriver.promise.controlFlow().getHistory();
    assertEquals(3, flowHistory.length);
    assertFrame('three', flowHistory[0]);
    assertFrame('two', flowHistory[1]);
    assertFrame('one', flowHistory[2]);
  });
  runAndExpectSuccess();
  assertFlowHistory('one', 'two', 'three');
  assertEquals(0, webdriver.promise.controlFlow().getHistory().length);
}

function testHistory_subtaskFailureIsIgnoredByErrback() {
  scheduleAction('one', function() {

    scheduleAction('two', function() {
      scheduleAction('three', throwStubError);
    }).thenCatch(goog.nullFunction);

    schedule('post error').then(function() {
      var flowHistory = webdriver.promise.controlFlow().getHistory();
      assertEquals(2, flowHistory.length);
      assertFrame('post error', flowHistory[0]);
      assertFrame('one', flowHistory[1]);
    });
  });
  runAndExpectSuccess();
  assertFlowHistory('one', 'two', 'three', 'post error');
  assertEquals(0, webdriver.promise.controlFlow().getHistory().length);
}

function assertFlowIs(flow) {
  assertEquals(flow, webdriver.promise.controlFlow());
}

function testOwningFlowIsActivatedForExecutingTasks() {
  var defaultFlow = webdriver.promise.controlFlow();

  webdriver.promise.createFlow(function(flow) {
    assertFlowIs(flow);

    defaultFlow.execute(function() {
      assertFlowIs(defaultFlow);
    });
  });

  runAndExpectSuccess();
  assertFlowIs(defaultFlow);
}

function testCreateFlowReturnsPromisePairedWithCreatedFlow() {
  var defaultFlow = webdriver.promise.controlFlow();

  var newFlow;
  webdriver.promise.createFlow(function(flow) {
    newFlow = flow;
    assertFlowIs(newFlow);
  }).then(function() {
    assertFlowIs(newFlow);
  });

  runAndExpectSuccess();
}

function testDeferredFactoriesCreateForActiveFlow() {
  var e = Error();
  var defaultFlow = webdriver.promise.controlFlow();
  webdriver.promise.fulfilled().then(function() {
    assertFlowIs(defaultFlow);
  });
  webdriver.promise.rejected(e).then(null, function(err) {
    assertEquals(e, err);
    assertFlowIs(defaultFlow);
  });
  webdriver.promise.defer().then(function() {
    assertFlowIs(defaultFlow);
  });

  var newFlow;
  webdriver.promise.createFlow(function(flow) {
    newFlow = flow;
    webdriver.promise.fulfilled().then(function() {
      assertFlowIs(flow);
    });
    webdriver.promise.rejected(e).then(null, function(err) {
      assertEquals(e, err);
      assertFlowIs(flow);
    });
    webdriver.promise.defer().then(function() {
      assertFlowIs(flow);
    });
  }).then(function() {
    assertFlowIs(newFlow);
  });

  runAndExpectSuccess();
}

function testFlowsSynchronizeWithThemselvesNotEachOther() {
  var defaultFlow = webdriver.promise.controlFlow();
  schedulePush('a', 'a');
  webdriver.promise.controlFlow().timeout(250);
  schedulePush('b', 'b');

  webdriver.promise.createFlow(function() {
    schedulePush('c', 'c');
    schedulePush('d', 'd');
  });

  runAndExpectSuccess();
  webdriver.test.testutil.assertMessages('a', 'c', 'd', 'b');
}

function testUnhandledErrorsAreReportedToTheOwningFlow() {
  var error1 = Error();
  var error2 = Error();
  var defaultFlow = webdriver.promise.controlFlow();

  var newFlow;
  webdriver.promise.createFlow(function(flow) {
    newFlow = flow;
    webdriver.promise.rejected(error1);

    defaultFlow.execute(function() {
      webdriver.promise.rejected(error2);
    });
  });

  flowTester.run();
  assertEquals(error2, flowTester.getFailure(defaultFlow));
  assertEquals(error1, flowTester.getFailure(newFlow));
}

function testCanSynchronizeFlowsByReturningPromiseFromOneToAnother() {
  var defaultFlow = webdriver.promise.controlFlow();
  schedulePush('a', 'a');
  webdriver.promise.controlFlow().timeout(250);
  schedulePush('b', 'b');

  webdriver.promise.createFlow(function() {
    schedulePush('c', 'c');
    scheduleAction('', function() {
      return defaultFlow.execute(function() {
        assertFlowIs(defaultFlow);
        return schedulePush('e', 'e');
      });
    });
    schedulePush('d', 'd');
  });

  runAndExpectSuccess();
  webdriver.test.testutil.assertMessages('a', 'c', 'b', 'e', 'd');
}

function testFramesWaitToCompleteForPendingRejections() {
  webdriver.promise.controlFlow().execute(function() {
    webdriver.promise.rejected(STUB_ERROR);
  });

  runAndExpectFailure(assertIsStubError);
}

function testSynchronizeErrorsPropagateToOuterFlow() {
  var defaultFlow = webdriver.promise.controlFlow();

  var newFlow;
  webdriver.promise.createFlow(function(flow) {
    newFlow = flow;
    return defaultFlow.execute(function() {
      webdriver.promise.rejected(STUB_ERROR);
    });
  });

  flowTester.run();
  assertIsStubError(flowTester.getFailure(defaultFlow));
  flowTester.verifySuccess(newFlow);  // Error was transferred to new flow.
}

function testFailsIfErrbackThrows() {
  webdriver.promise.rejected('').then(null, throwStubError);
  runAndExpectFailure(assertIsStubError);
}

function testFailsIfCallbackReturnsRejectedPromise() {
  webdriver.promise.fulfilled().then(function() {
    return webdriver.promise.rejected(STUB_ERROR);
  });
  runAndExpectFailure(assertIsStubError);
}

function testAbortsFrameIfTaskFails() {
  webdriver.promise.fulfilled().then(function() {
    webdriver.promise.controlFlow().execute(throwStubError);
  });
  runAndExpectFailure(assertIsStubError);
}

function testAbortsFramePromisedChainedFromTaskIsNotHandled() {
  webdriver.promise.fulfilled().then(function() {
    webdriver.promise.controlFlow().execute(goog.nullFunction).
        then(throwStubError);
  });
  runAndExpectFailure(assertIsStubError);
}

function testTrapsChainedUnhandledRejectionsWithinAFrame() {
  var pair = callbackPair(null, assertIsStubError);
  webdriver.promise.fulfilled().then(function() {
    webdriver.promise.controlFlow().execute(goog.nullFunction).
        then(throwStubError);
  }).then(pair.callback, pair.errback);

  runAndExpectSuccess();
  pair.assertErrback();
}


function testCancelsRemainingTasksIfFrameThrowsDuringScheduling() {
  var task1, task2;
  var pair = callbackPair(null, assertIsStubError);
  var flow = webdriver.promise.controlFlow();
  flow.execute(function() {
    task1 = flow.execute(goog.nullFunction);
    task2 = flow.execute(goog.nullFunction);
    throw STUB_ERROR;
  }).then(pair.callback, pair.errback);

  runAndExpectSuccess();
  pair.assertErrback();

  assertFalse(task1.isPending());
  pair = callbackPair();
  task1.then(pair.callback, pair.errback);
  pair.assertErrback();

  assertFalse(task2.isPending());
  pair = callbackPair();
  task2.then(pair.callback, pair.errback);
  pair.assertErrback();
}

function testCancelsRemainingTasksInFrameIfATaskFails() {
  var task;
  var pair = callbackPair(null, assertIsStubError);
  var flow = webdriver.promise.controlFlow();
  flow.execute(function() {
    flow.execute(throwStubError);
    task = flow.execute(goog.nullFunction);
  }).then(pair.callback, pair.errback);

  runAndExpectSuccess();
  pair.assertErrback();

  assertFalse(task.isPending());
  pair = callbackPair();
  task.then(pair.callback, pair.errback);
  pair.assertErrback();
}

function testAnnotatesRejectedPromiseErrorsWithFlowState() {
  var error = Error('original message');
  var originalStack = webdriver.stacktrace.format(error).stack;

  var pair = callbackPair(null, function(e) {
    assertEquals(error, e);
    assertEquals('original message', e.message);
    assertTrue(
      'Expected to start with: ' + originalStack,
      goog.string.startsWith(e.stack, originalStack));

    var parts = e.stack.split('\n==== async task ====\n');
    assertEquals(2, parts.length);
    assertEquals(originalStack, parts[0]);
  });

  webdriver.promise.createFlow(function(flow) {
    var d = webdriver.promise.defer();
    d.reject(error);
    d.then(pair.callback, pair.errback);
  });

  runAndExpectSuccess();
  pair.assertErrback();
}

function testAnnotatesChainedErrors() {
  var error = Error('original message');
  var originalStack = webdriver.stacktrace.format(error).stack;

  var pair = callbackPair(null, function(e) {
    assertEquals(error, e);
    assertEquals('original message', e.message);
    assertTrue(
      'Expected to start with: ' + originalStack,
      goog.string.startsWith(e.stack, originalStack));

    var parts = e.stack.split('\n==== async task ====\n');
    assertEquals(2, parts.length);
    assertEquals(originalStack, parts[0]);
  });

  webdriver.promise.createFlow(function(flow) {
    var rejected = webdriver.promise.rejected(error);
    webdriver.promise.fulfilled(rejected).
        then(pair.callback, pair.errback);
  });

  runAndExpectSuccess();
  pair.assertErrback();
}

function testAnnotatesRejectedPromiseErrorsWithFlowState_taskErrorBubblesUp() {
  var error = Error('original message');
  var originalStack = webdriver.stacktrace.format(error).stack;
  var pair = callbackPair(null, function(e) {
    assertEquals(error, e);
    assertEquals('original message', e.message);
    assertTrue(
      'Expected to start with: ' + originalStack,
      goog.string.startsWith(e.stack, originalStack));

    var parts = e.stack.split('\n==== async task ====\n');
    assertEquals(3, parts.length);
    assertEquals(originalStack, parts[0]);
  });

  webdriver.promise.createFlow(function(flow) {
    flow.execute(function() { throw error; });
  }).then(pair.callback, pair.errback);

  runAndExpectSuccess();
  pair.assertErrback();
}

function testDoesNotAnnotatedRejectedPromisesIfGivenNonErrorValue() {
  var error = {};

  var pair = callbackPair(null, function(e) {
    assertEquals(error, e);
    for (var val in error) {
      fail('Did not expect error to be modified');
    }
  });

  webdriver.promise.createFlow(function(flow) {
    var d = webdriver.promise.defer();
    d.reject(error);
    d.then(pair.callback, pair.errback);
  });

  runAndExpectSuccess();
  pair.assertErrback();
}

function testDoesNotModifyRejectionErrorIfPromiseNotInsideAFlow() {
  var error = Error('original message');
  var originalStack = error.stack;
  var originalStr = error.toString();

  var pair = callbackPair(null, function(e) {
    assertEquals(error, e);
    assertEquals('original message', e.message);
    assertEquals(originalStack, e.stack);
    assertEquals(originalStr, e.toString());
  });

  webdriver.promise.rejected(error).then(pair.callback, pair.errback);
  pair.assertErrback();
}
