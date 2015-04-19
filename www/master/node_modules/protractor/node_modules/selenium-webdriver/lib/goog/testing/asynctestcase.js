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
// All Rights Reserved.

/**
 * @fileoverview A class representing a set of test functions that use
 * asynchronous functions that cannot be meaningfully mocked.
 *
 * To create a Google-compatable JsUnit test using this test case, put the
 * following snippet in your test:
 *
 *   var asyncTestCase = goog.testing.AsyncTestCase.createAndInstall();
 *
 * To make the test runner wait for your asynchronous behaviour, use:
 *
 *   asyncTestCase.waitForAsync('Waiting for xhr to respond');
 *
 * The next test will not start until the following call is made, or a
 * timeout occurs:
 *
 *   asyncTestCase.continueTesting();
 *
 * There does NOT need to be a 1:1 mapping of waitForAsync calls and
 * continueTesting calls. The next test will be run after a single call to
 * continueTesting is made, as long as there is no subsequent call to
 * waitForAsync in the same thread.
 *
 * Example:
 *   // Returning here would cause the next test to be run.
 *   asyncTestCase.waitForAsync('description 1');
 *   // Returning here would *not* cause the next test to be run.
 *   // Only effect of additional waitForAsync() calls is an updated
 *   // description in the case of a timeout.
 *   asyncTestCase.waitForAsync('updated description');
 *   asyncTestCase.continueTesting();
 *   // Returning here would cause the next test to be run.
 *   asyncTestCase.waitForAsync('just kidding, still running.');
 *   // Returning here would *not* cause the next test to be run.
 *
 * The test runner can also be made to wait for more than one asynchronous
 * event with:
 *
 *   asyncTestCase.waitForSignals(n);
 *
 * The next test will not start until asyncTestCase.signal() is called n times,
 * or the test step timeout is exceeded.
 *
 * This class supports asynchronous behaviour in all test functions except for
 * tearDownPage. If such support is needed, it can be added.
 *
 * Example Usage:
 *
 *   var asyncTestCase = goog.testing.AsyncTestCase.createAndInstall();
 *   // Optionally, set a longer-than-normal step timeout.
 *   asyncTestCase.stepTimeout = 30 * 1000;
 *
 *   function testSetTimeout() {
 *     var step = 0;
 *     function stepCallback() {
 *       step++;
 *       switch (step) {
 *         case 1:
 *           var startTime = goog.now();
 *           asyncTestCase.waitForAsync('step 1');
 *           window.setTimeout(stepCallback, 100);
 *           break;
 *         case 2:
 *           assertTrue('Timeout fired too soon',
 *               goog.now() - startTime >= 100);
 *           asyncTestCase.waitForAsync('step 2');
 *           window.setTimeout(stepCallback, 100);
 *           break;
 *         case 3:
 *           assertTrue('Timeout fired too soon',
 *               goog.now() - startTime >= 200);
 *           asyncTestCase.continueTesting();
 *           break;
 *         default:
 *           fail('Unexpected call to stepCallback');
 *       }
 *     }
 *     stepCallback();
 *   }
 *
 * Known Issues:
 *   IE7 Exceptions:
 *     As the failingtest.html will show, it appears as though ie7 does not
 *     propagate an exception past a function called using the func.call()
 *     syntax. This causes case 3 of the failing tests (exceptions) to show up
 *     as timeouts in IE.
 *   window.onerror:
 *     This seems to catch errors only in ff2/ff3. It does not work in Safari or
 *     IE7. The consequence of this is that exceptions that would have been
 *     caught by window.onerror show up as timeouts.
 *
 * @author agrieve@google.com (Andrew Grieve)
 */

goog.provide('goog.testing.AsyncTestCase');
goog.provide('goog.testing.AsyncTestCase.ControlBreakingException');

goog.require('goog.testing.TestCase');
goog.require('goog.testing.TestCase.Test');
goog.require('goog.testing.asserts');



/**
 * A test case that is capable of running tests the contain asynchronous logic.
 * @param {string=} opt_name A descriptive name for the test case.
 * @extends {goog.testing.TestCase}
 * @constructor
 */
goog.testing.AsyncTestCase = function(opt_name) {
  goog.testing.TestCase.call(this, opt_name);
};
goog.inherits(goog.testing.AsyncTestCase, goog.testing.TestCase);


/**
 * Represents result of top stack function call.
 * @typedef {{controlBreakingExceptionThrown: boolean, message: string}}
 * @private
 */
goog.testing.AsyncTestCase.TopStackFuncResult_;



/**
 * An exception class used solely for control flow.
 * @param {string=} opt_message Error message.
 * @constructor
 * @final
 */
goog.testing.AsyncTestCase.ControlBreakingException = function(opt_message) {
  /**
   * The exception message.
   * @type {string}
   */
  this.message = opt_message || '';
};


/**
 * Return value for .toString().
 * @type {string}
 */
goog.testing.AsyncTestCase.ControlBreakingException.TO_STRING =
    '[AsyncTestCase.ControlBreakingException]';


/**
 * Marks this object as a ControlBreakingException
 * @type {boolean}
 */
goog.testing.AsyncTestCase.ControlBreakingException.prototype.
    isControlBreakingException = true;


/** @override */
goog.testing.AsyncTestCase.ControlBreakingException.prototype.toString =
    function() {
  // This shows up in the console when the exception is not caught.
  return goog.testing.AsyncTestCase.ControlBreakingException.TO_STRING;
};


/**
 * How long to wait for a single step of a test to complete in milliseconds.
 * A step starts when a call to waitForAsync() is made.
 * @type {number}
 */
goog.testing.AsyncTestCase.prototype.stepTimeout = 1000;


/**
 * How long to wait after a failed test before moving onto the next one.
 * The purpose of this is to allow any pending async callbacks from the failing
 * test to finish up and not cause the next test to fail.
 * @type {number}
 */
goog.testing.AsyncTestCase.prototype.timeToSleepAfterFailure = 500;


/**
 * Turn on extra logging to help debug failing async. tests.
 * @type {boolean}
 * @private
 */
goog.testing.AsyncTestCase.prototype.enableDebugLogs_ = false;


/**
 * A reference to the original asserts.js assert_() function.
 * @private
 */
goog.testing.AsyncTestCase.prototype.origAssert_;


/**
 * A reference to the original asserts.js fail() function.
 * @private
 */
goog.testing.AsyncTestCase.prototype.origFail_;


/**
 * A reference to the original window.onerror function.
 * @type {Function|undefined}
 * @private
 */
goog.testing.AsyncTestCase.prototype.origOnError_;


/**
 * The stage of the test we are currently on.
 * @type {Function|undefined}}
 * @private
 */
goog.testing.AsyncTestCase.prototype.curStepFunc_;


/**
 * The name of the stage of the test we are currently on.
 * @type {string}
 * @private
 */
goog.testing.AsyncTestCase.prototype.curStepName_ = '';


/**
 * The stage of the test we should run next.
 * @type {Function|undefined}
 * @private
 */
goog.testing.AsyncTestCase.prototype.nextStepFunc;


/**
 * The name of the stage of the test we should run next.
 * @type {string}
 * @private
 */
goog.testing.AsyncTestCase.prototype.nextStepName_ = '';


/**
 * The handle to the current setTimeout timer.
 * @type {number}
 * @private
 */
goog.testing.AsyncTestCase.prototype.timeoutHandle_ = 0;


/**
 * Marks if the cleanUp() function has been called for the currently running
 * test.
 * @type {boolean}
 * @private
 */
goog.testing.AsyncTestCase.prototype.cleanedUp_ = false;


/**
 * The currently active test.
 * @type {goog.testing.TestCase.Test|undefined}
 * @protected
 */
goog.testing.AsyncTestCase.prototype.activeTest;


/**
 * A flag to prevent recursive exception handling.
 * @type {boolean}
 * @private
 */
goog.testing.AsyncTestCase.prototype.inException_ = false;


/**
 * Flag used to determine if we can move to the next step in the testing loop.
 * @type {boolean}
 * @private
 */
goog.testing.AsyncTestCase.prototype.isReady_ = true;


/**
 * Number of signals to wait for before continuing testing when waitForSignals
 * is used.
 * @type {number}
 * @private
 */
goog.testing.AsyncTestCase.prototype.expectedSignalCount_ = 0;


/**
 * Number of signals received.
 * @type {number}
 * @private
 */
goog.testing.AsyncTestCase.prototype.receivedSignalCount_ = 0;


/**
 * Flag that tells us if there is a function in the call stack that will make
 * a call to pump_().
 * @type {boolean}
 * @private
 */
goog.testing.AsyncTestCase.prototype.returnWillPump_ = false;


/**
 * The number of times we have thrown a ControlBreakingException so that we
 * know not to complain in our window.onerror handler. In Webkit, window.onerror
 * is not supported, and so this counter will keep going up but we won't care
 * about it.
 * @type {number}
 * @private
 */
goog.testing.AsyncTestCase.prototype.numControlExceptionsExpected_ = 0;


/**
 * The current step name.
 * @return {!string} Step name.
 * @protected
 */
goog.testing.AsyncTestCase.prototype.getCurrentStepName = function() {
  return this.curStepName_;
};


/**
 * Preferred way of creating an AsyncTestCase. Creates one and initializes it
 * with the G_testRunner.
 * @param {string=} opt_name A descriptive name for the test case.
 * @return {!goog.testing.AsyncTestCase} The created AsyncTestCase.
 */
goog.testing.AsyncTestCase.createAndInstall = function(opt_name) {
  var asyncTestCase = new goog.testing.AsyncTestCase(opt_name);
  goog.testing.TestCase.initializeTestRunner(asyncTestCase);
  return asyncTestCase;
};


/**
 * Informs the testcase not to continue to the next step in the test cycle
 * until continueTesting is called.
 * @param {string=} opt_name A description of what we are waiting for.
 */
goog.testing.AsyncTestCase.prototype.waitForAsync = function(opt_name) {
  this.isReady_ = false;
  this.curStepName_ = opt_name || this.curStepName_;

  // Reset the timer that tracks if the async test takes too long.
  this.stopTimeoutTimer_();
  this.startTimeoutTimer_();
};


/**
 * Continue with the next step in the test cycle.
 */
goog.testing.AsyncTestCase.prototype.continueTesting = function() {
  if (this.receivedSignalCount_ < this.expectedSignalCount_) {
    var remaining = this.expectedSignalCount_ - this.receivedSignalCount_;
    throw Error('Still waiting for ' + remaining + ' signals.');
  }
  this.endCurrentStep_();
};


/**
 * Ends the current test step and queues the next test step to run.
 * @private
 */
goog.testing.AsyncTestCase.prototype.endCurrentStep_ = function() {
  if (!this.isReady_) {
    // We are a potential entry point, so we pump.
    this.isReady_ = true;
    this.stopTimeoutTimer_();
    // Run this in a setTimeout so that the caller has a chance to call
    // waitForAsync() again before we continue.
    this.timeout(goog.bind(this.pump_, this, null), 0);
  }
};


/**
 * Informs the testcase not to continue to the next step in the test cycle
 * until signal is called the specified number of times. Within a test, this
 * function behaves additively if called multiple times; the number of signals
 * to wait for will be the sum of all expected number of signals this function
 * was called with.
 * @param {number} times The number of signals to receive before
 *    continuing testing.
 * @param {string=} opt_name A description of what we are waiting for.
 */
goog.testing.AsyncTestCase.prototype.waitForSignals =
    function(times, opt_name) {
  this.expectedSignalCount_ += times;
  if (this.receivedSignalCount_ < this.expectedSignalCount_) {
    this.waitForAsync(opt_name);
  }
};


/**
 * Signals once to continue with the test. If this is the last signal that the
 * test was waiting on, call continueTesting.
 */
goog.testing.AsyncTestCase.prototype.signal = function() {
  if (++this.receivedSignalCount_ === this.expectedSignalCount_ &&
      this.expectedSignalCount_ > 0) {
    this.endCurrentStep_();
  }
};


/**
 * Handles an exception thrown by a test.
 * @param {*=} opt_e The exception object associated with the failure
 *     or a string.
 * @throws Always throws a ControlBreakingException.
 */
goog.testing.AsyncTestCase.prototype.doAsyncError = function(opt_e) {
  // If we've caught an exception that we threw, then just pass it along. This
  // can happen if doAsyncError() was called from a call to assert and then
  // again by pump_().
  if (opt_e && opt_e.isControlBreakingException) {
    throw opt_e;
  }

  // Prevent another timeout error from triggering for this test step.
  this.stopTimeoutTimer_();

  // doError() uses test.name. Here, we create a dummy test and give it a more
  // helpful name based on the step we're currently on.
  var fakeTestObj = new goog.testing.TestCase.Test(this.curStepName_,
                                                   goog.nullFunction);
  if (this.activeTest) {
    fakeTestObj.name = this.activeTest.name + ' [' + fakeTestObj.name + ']';
  }

  if (this.activeTest) {
    // Note: if the test has an error, and then tearDown has an error, they will
    // both be reported.
    this.doError(fakeTestObj, opt_e);
  } else {
    this.exceptionBeforeTest = opt_e;
  }

  // This is a potential entry point, so we pump. We also add in a bit of a
  // delay to try and prevent any async behavior from the failed test from
  // causing the next test to fail.
  this.timeout(goog.bind(this.pump_, this, this.doAsyncErrorTearDown_),
      this.timeToSleepAfterFailure);

  // We just caught an exception, so we do not want the code above us on the
  // stack to continue executing. If pump_ is in our call-stack, then it will
  // batch together multiple errors, so we only increment the count if pump_ is
  // not in the stack and let pump_ increment the count when it batches them.
  if (!this.returnWillPump_) {
    this.numControlExceptionsExpected_ += 1;
    this.dbgLog_('doAsynError: numControlExceptionsExpected_ = ' +
        this.numControlExceptionsExpected_ + ' and throwing exception.');
  }

  // Copy the error message to ControlBreakingException.
  var message = '';
  if (typeof opt_e == 'string') {
    message = opt_e;
  } else if (opt_e && opt_e.message) {
    message = opt_e.message;
  }
  throw new goog.testing.AsyncTestCase.ControlBreakingException(message);
};


/**
 * Sets up the test page and then waits until the test case has been marked
 * as ready before executing the tests.
 * @override
 */
goog.testing.AsyncTestCase.prototype.runTests = function() {
  this.hookAssert_();
  this.hookOnError_();

  this.setNextStep_(this.doSetUpPage_, 'setUpPage');
  // We are an entry point, so we pump.
  this.pump_();
};


/**
 * Starts the tests.
 * @override
 */
goog.testing.AsyncTestCase.prototype.cycleTests = function() {
  // We are an entry point, so we pump.
  this.saveMessage('Start');
  this.setNextStep_(this.doIteration_, 'doIteration');
  this.pump_();
};


/**
 * Finalizes the test case, called when the tests have finished executing.
 * @override
 */
goog.testing.AsyncTestCase.prototype.finalize = function() {
  this.unhookAll_();
  this.setNextStep_(null, 'finalized');
  goog.testing.AsyncTestCase.superClass_.finalize.call(this);
};


/**
 * Enables verbose logging of what is happening inside of the AsyncTestCase.
 */
goog.testing.AsyncTestCase.prototype.enableDebugLogging = function() {
  this.enableDebugLogs_ = true;
};


/**
 * Logs the given debug message to the console (when enabled).
 * @param {string} message The message to log.
 * @private
 */
goog.testing.AsyncTestCase.prototype.dbgLog_ = function(message) {
  if (this.enableDebugLogs_) {
    this.log('AsyncTestCase - ' + message);
  }
};


/**
 * Wraps doAsyncError() for when we are sure that the test runner has no user
 * code above it in the stack.
 * @param {string|Error=} opt_e The exception object associated with the
 *     failure or a string.
 * @private
 */
goog.testing.AsyncTestCase.prototype.doTopOfStackAsyncError_ =
    function(opt_e) {
  /** @preserveTry */
  try {
    this.doAsyncError(opt_e);
  } catch (e) {
    // We know that we are on the top of the stack, so there is no need to
    // throw this exception in this case.
    if (e.isControlBreakingException) {
      this.numControlExceptionsExpected_ -= 1;
      this.dbgLog_('doTopOfStackAsyncError_: numControlExceptionsExpected_ = ' +
          this.numControlExceptionsExpected_ + ' and catching exception.');
    } else {
      throw e;
    }
  }
};


/**
 * Calls the tearDown function, catching any errors, and then moves on to
 * the next step in the testing cycle.
 * @private
 */
goog.testing.AsyncTestCase.prototype.doAsyncErrorTearDown_ = function() {
  if (this.inException_) {
    // We get here if tearDown is throwing the error.
    // Upon calling continueTesting, the inline function 'doAsyncError' (set
    // below) is run.
    this.endCurrentStep_();
  } else {
    this.inException_ = true;
    this.isReady_ = true;

    // The continue point is different depending on if the error happened in
    // setUpPage() or in setUp()/test*()/tearDown().
    var stepFuncAfterError = this.nextStepFunc_;
    var stepNameAfterError = 'TestCase.execute (after error)';
    if (this.activeTest) {
      stepFuncAfterError = this.doIteration_;
      stepNameAfterError = 'doIteration (after error)';
    }

    // We must set the next step before calling tearDown.
    this.setNextStep_(function() {
      this.inException_ = false;
      // This is null when an error happens in setUpPage.
      this.setNextStep_(stepFuncAfterError, stepNameAfterError);
    }, 'doAsyncError');

    // Call the test's tearDown().
    if (!this.cleanedUp_) {
      this.cleanedUp_ = true;
      this.tearDown();
    }
  }
};


/**
 * Replaces the asserts.js assert_() and fail() functions with a wrappers to
 * catch the exceptions.
 * @private
 */
goog.testing.AsyncTestCase.prototype.hookAssert_ = function() {
  if (!this.origAssert_) {
    this.origAssert_ = _assert;
    this.origFail_ = fail;
    var self = this;
    _assert = function() {
      /** @preserveTry */
      try {
        self.origAssert_.apply(this, arguments);
      } catch (e) {
        self.dbgLog_('Wrapping failed assert()');
        self.doAsyncError(e);
      }
    };
    fail = function() {
      /** @preserveTry */
      try {
        self.origFail_.apply(this, arguments);
      } catch (e) {
        self.dbgLog_('Wrapping fail()');
        self.doAsyncError(e);
      }
    };
  }
};


/**
 * Sets a window.onerror handler for catching exceptions that happen in async
 * callbacks. Note that as of Safari 3.1, Safari does not support this.
 * @private
 */
goog.testing.AsyncTestCase.prototype.hookOnError_ = function() {
  if (!this.origOnError_) {
    this.origOnError_ = window.onerror;
    var self = this;
    window.onerror = function(error, url, line) {
      // Ignore exceptions that we threw on purpose.
      var cbe =
          goog.testing.AsyncTestCase.ControlBreakingException.TO_STRING;
      if (String(error).indexOf(cbe) != -1 &&
          self.numControlExceptionsExpected_) {
        self.numControlExceptionsExpected_ -= 1;
        self.dbgLog_('window.onerror: numControlExceptionsExpected_ = ' +
            self.numControlExceptionsExpected_ + ' and ignoring exception. ' +
            error);
        // Tell the browser not to compain about the error.
        return true;
      } else {
        self.dbgLog_('window.onerror caught exception.');
        var message = error + '\nURL: ' + url + '\nLine: ' + line;
        self.doTopOfStackAsyncError_(message);
        // Tell the browser to complain about the error.
        return false;
      }
    };
  }
};


/**
 * Unhooks window.onerror and _assert.
 * @private
 */
goog.testing.AsyncTestCase.prototype.unhookAll_ = function() {
  if (this.origOnError_) {
    window.onerror = this.origOnError_;
    this.origOnError_ = null;
    _assert = this.origAssert_;
    this.origAssert_ = null;
    fail = this.origFail_;
    this.origFail_ = null;
  }
};


/**
 * Enables the timeout timer. This timer fires unless continueTesting is
 * called.
 * @private
 */
goog.testing.AsyncTestCase.prototype.startTimeoutTimer_ = function() {
  if (!this.timeoutHandle_ && this.stepTimeout > 0) {
    this.timeoutHandle_ = this.timeout(goog.bind(function() {
      this.dbgLog_('Timeout timer fired with id ' + this.timeoutHandle_);
      this.timeoutHandle_ = 0;

      this.doTopOfStackAsyncError_('Timed out while waiting for ' +
          'continueTesting() to be called.');
    }, this, null), this.stepTimeout);
    this.dbgLog_('Started timeout timer with id ' + this.timeoutHandle_);
  }
};


/**
 * Disables the timeout timer.
 * @private
 */
goog.testing.AsyncTestCase.prototype.stopTimeoutTimer_ = function() {
  if (this.timeoutHandle_) {
    this.dbgLog_('Clearing timeout timer with id ' + this.timeoutHandle_);
    this.clearTimeout(this.timeoutHandle_);
    this.timeoutHandle_ = 0;
  }
};


/**
 * Sets the next function to call in our sequence of async callbacks.
 * @param {Function} func The function that executes the next step.
 * @param {string} name A description of the next step.
 * @private
 */
goog.testing.AsyncTestCase.prototype.setNextStep_ = function(func, name) {
  this.nextStepFunc_ = func && goog.bind(func, this);
  this.nextStepName_ = name;
};


/**
 * Calls the given function, redirecting any exceptions to doAsyncError.
 * @param {Function} func The function to call.
 * @return {!goog.testing.AsyncTestCase.TopStackFuncResult_} Returns a
 * TopStackFuncResult_.
 * @private
 */
goog.testing.AsyncTestCase.prototype.callTopOfStackFunc_ = function(func) {
  /** @preserveTry */
  try {
    func.call(this);
    return {controlBreakingExceptionThrown: false, message: ''};
  } catch (e) {
    this.dbgLog_('Caught exception in callTopOfStackFunc_');
    /** @preserveTry */
    try {
      this.doAsyncError(e);
      return {controlBreakingExceptionThrown: false, message: ''};
    } catch (e2) {
      if (!e2.isControlBreakingException) {
        throw e2;
      }
      return {controlBreakingExceptionThrown: true, message: e2.message};
    }
  }
};


/**
 * Calls the next callback when the isReady_ flag is true.
 * @param {Function=} opt_doFirst A function to call before pumping.
 * @private
 * @throws Throws a ControlBreakingException if there were any failing steps.
 */
goog.testing.AsyncTestCase.prototype.pump_ = function(opt_doFirst) {
  // If this function is already above us in the call-stack, then we should
  // return rather than pumping in order to minimize call-stack depth.
  if (!this.returnWillPump_) {
    this.setBatchTime(this.now());
    this.returnWillPump_ = true;
    var topFuncResult = {};

    if (opt_doFirst) {
      topFuncResult = this.callTopOfStackFunc_(opt_doFirst);
    }
    // Note: we don't check for this.running here because it is not set to true
    // while executing setUpPage and tearDownPage.
    // Also, if isReady_ is false, then one of two things will happen:
    // 1. Our timeout callback will be called.
    // 2. The tests will call continueTesting(), which will call pump_() again.
    while (this.isReady_ && this.nextStepFunc_ &&
        !topFuncResult.controlBreakingExceptionThrown) {
      this.curStepFunc_ = this.nextStepFunc_;
      this.curStepName_ = this.nextStepName_;
      this.nextStepFunc_ = null;
      this.nextStepName_ = '';

      this.dbgLog_('Performing step: ' + this.curStepName_);
      topFuncResult =
          this.callTopOfStackFunc_(/** @type {Function} */(this.curStepFunc_));

      // If the max run time is exceeded call this function again async so as
      // not to block the browser.
      var delta = this.now() - this.getBatchTime();
      if (delta > goog.testing.TestCase.maxRunTime &&
          !topFuncResult.controlBreakingExceptionThrown) {
        this.saveMessage('Breaking async');
        var self = this;
        this.timeout(function() { self.pump_(); }, 100);
        break;
      }
    }
    this.returnWillPump_ = false;
  } else if (opt_doFirst) {
    opt_doFirst.call(this);
  }
};


/**
 * Sets up the test page and then waits untill the test case has been marked
 * as ready before executing the tests.
 * @private
 */
goog.testing.AsyncTestCase.prototype.doSetUpPage_ = function() {
  this.setNextStep_(this.execute, 'TestCase.execute');
  this.setUpPage();
};


/**
 * Step 1: Move to the next test.
 * @private
 */
goog.testing.AsyncTestCase.prototype.doIteration_ = function() {
  this.expectedSignalCount_ = 0;
  this.receivedSignalCount_ = 0;
  this.activeTest = this.next();
  if (this.activeTest && this.running) {
    this.result_.runCount++;
    // If this test should be marked as having failed, doIteration will go
    // straight to the next test.
    if (this.maybeFailTestEarly(this.activeTest)) {
      this.setNextStep_(this.doIteration_, 'doIteration');
    } else {
      this.setNextStep_(this.doSetUp_, 'setUp');
    }
  } else {
    // All tests done.
    this.finalize();
  }
};


/**
 * Step 2: Call setUp().
 * @private
 */
goog.testing.AsyncTestCase.prototype.doSetUp_ = function() {
  this.log('Running test: ' + this.activeTest.name);
  this.cleanedUp_ = false;
  this.setNextStep_(this.doExecute_, this.activeTest.name);
  this.setUp();
};


/**
 * Step 3: Call test.execute().
 * @private
 */
goog.testing.AsyncTestCase.prototype.doExecute_ = function() {
  this.setNextStep_(this.doTearDown_, 'tearDown');
  this.activeTest.execute();
};


/**
 * Step 4: Call tearDown().
 * @private
 */
goog.testing.AsyncTestCase.prototype.doTearDown_ = function() {
  this.cleanedUp_ = true;
  this.setNextStep_(this.doNext_, 'doNext');
  this.tearDown();
};


/**
 * Step 5: Call doSuccess()
 * @private
 */
goog.testing.AsyncTestCase.prototype.doNext_ = function() {
  this.setNextStep_(this.doIteration_, 'doIteration');
  this.doSuccess(/** @type {goog.testing.TestCase.Test} */(this.activeTest));
};
