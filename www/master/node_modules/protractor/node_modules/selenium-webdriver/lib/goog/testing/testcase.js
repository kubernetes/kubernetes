// Copyright 2007 The Closure Library Authors. All Rights Reserved.
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
 * @fileoverview A class representing a set of test functions to be run.
 *
 * Testing code should not have dependencies outside of goog.testing so as to
 * reduce the chance of masking missing dependencies.
 *
 * This file does not compile correctly with --collapse_properties. Use
 * --property_renaming=ALL_UNQUOTED instead.
 *
 */

goog.provide('goog.testing.TestCase');
goog.provide('goog.testing.TestCase.Error');
goog.provide('goog.testing.TestCase.Order');
goog.provide('goog.testing.TestCase.Result');
goog.provide('goog.testing.TestCase.Test');

goog.require('goog.object');
goog.require('goog.testing.asserts');
goog.require('goog.testing.stacktrace');



/**
 * A class representing a JsUnit test case.  A TestCase is made up of a number
 * of test functions which can be run.  Individual test cases can override the
 * following functions to set up their test environment:
 *   - runTests - completely override the test's runner
 *   - setUpPage - called before any of the test functions are run
 *   - tearDownPage - called after all tests are finished
 *   - setUp - called before each of the test functions
 *   - tearDown - called after each of the test functions
 *   - shouldRunTests - called before a test run, all tests are skipped if it
 *                      returns false.  Can be used to disable tests on browsers
 *                      where they aren't expected to pass.
 *
 * Use {@link #autoDiscoverLifecycle} and {@link #autoDiscoverTests}
 *
 * @param {string=} opt_name The name of the test case, defaults to
 *     'Untitled Test Case'.
 * @constructor
 */
goog.testing.TestCase = function(opt_name) {
  /**
   * A name for the test case.
   * @type {string}
   * @private
   */
  this.name_ = opt_name || 'Untitled Test Case';

  /**
   * Array of test functions that can be executed.
   * @type {!Array.<!goog.testing.TestCase.Test>}
   * @private
   */
  this.tests_ = [];

  /**
   * Set of test names and/or indices to execute, or null if all tests should
   * be executed.
   *
   * Indices are included to allow automation tools to run a subset of the
   * tests without knowing the exact contents of the test file.
   *
   * Indices should only be used with SORTED ordering.
   *
   * Example valid values:
   * <ul>
   * <li>[testName]
   * <li>[testName1, testName2]
   * <li>[2] - will run the 3rd test in the order specified
   * <li>[1,3,5]
   * <li>[testName1, testName2, 3, 5] - will work
   * <ul>
   * @type {Object}
   * @private
   */
  this.testsToRun_ = null;

  var search = '';
  if (goog.global.location) {
    search = goog.global.location.search;
  }

  // Parse the 'runTests' query parameter into a set of test names and/or
  // test indices.
  var runTestsMatch = search.match(/(?:\?|&)runTests=([^?&]+)/i);
  if (runTestsMatch) {
    this.testsToRun_ = {};
    var arr = runTestsMatch[1].split(',');
    for (var i = 0, len = arr.length; i < len; i++) {
      this.testsToRun_[arr[i]] = 1;
    }
  }

  // Checks the URL for a valid order param.
  var orderMatch = search.match(/(?:\?|&)order=(natural|random|sorted)/i);
  if (orderMatch) {
    this.order = orderMatch[1];
  }

  /**
   * Object used to encapsulate the test results.
   * @type {goog.testing.TestCase.Result}
   * @protected
   * @suppress {underscore|visibility}
   */
  this.result_ = new goog.testing.TestCase.Result(this);

  // This silences a compiler warning from the legacy property check, which
  // is deprecated. It idly writes to testRunner properties that are used
  // in this file.
  var testRunnerMethods = {isFinished: true, hasErrors: true};
};


/**
 * The order to run the auto-discovered tests.
 * @enum {string}
 */
goog.testing.TestCase.Order = {
  /**
   * This is browser dependent and known to be different in FF and Safari
   * compared to others.
   */
  NATURAL: 'natural',

  /** Random order. */
  RANDOM: 'random',

  /** Sorted based on the name. */
  SORTED: 'sorted'
};


/**
 * @return {string} The name of the test.
 */
goog.testing.TestCase.prototype.getName = function() {
  return this.name_;
};


/**
 * The maximum amount of time that the test can run before we force it to be
 * async.  This prevents the test runner from blocking the browser and
 * potentially hurting the Selenium test harness.
 * @type {number}
 */
goog.testing.TestCase.maxRunTime = 200;


/**
 * The order to run the auto-discovered tests in.
 * @type {string}
 */
goog.testing.TestCase.prototype.order = goog.testing.TestCase.Order.SORTED;


/**
 * Save a reference to {@code window.setTimeout}, so any code that overrides the
 * default behavior (the MockClock, for example) doesn't affect our runner.
 * @type {function((Function|string), number, *=): number}
 * @private
 */
goog.testing.TestCase.protectedSetTimeout_ = goog.global.setTimeout;


/**
 * Save a reference to {@code window.clearTimeout}, so any code that overrides
 * the default behavior (e.g. MockClock) doesn't affect our runner.
 * @type {function((null|number|undefined)): void}
 * @private
 */
goog.testing.TestCase.protectedClearTimeout_ = goog.global.clearTimeout;


/**
 * Save a reference to {@code window.Date}, so any code that overrides
 * the default behavior doesn't affect our runner.
 * @type {function(new: Date)}
 * @private
 */
goog.testing.TestCase.protectedDate_ = Date;


/**
 * Saved string referencing goog.global.setTimeout's string serialization.  IE
 * sometimes fails to uphold equality for setTimeout, but the string version
 * stays the same.
 * @type {string}
 * @private
 */
goog.testing.TestCase.setTimeoutAsString_ = String(goog.global.setTimeout);


/**
 * TODO(user) replace this with prototype.currentTest.
 * Name of the current test that is running, or null if none is running.
 * @type {?string}
 */
goog.testing.TestCase.currentTestName = null;


/**
 * Avoid a dependency on goog.userAgent and keep our own reference of whether
 * the browser is IE.
 * @type {boolean}
 */
goog.testing.TestCase.IS_IE = typeof opera == 'undefined' &&
    !!goog.global.navigator &&
    goog.global.navigator.userAgent.indexOf('MSIE') != -1;


/**
 * Exception object that was detected before a test runs.
 * @type {*}
 * @protected
 */
goog.testing.TestCase.prototype.exceptionBeforeTest;


/**
 * Whether the test case has ever tried to execute.
 * @type {boolean}
 */
goog.testing.TestCase.prototype.started = false;


/**
 * Whether the test case is running.
 * @type {boolean}
 */
goog.testing.TestCase.prototype.running = false;


/**
 * Timestamp for when the test was started.
 * @type {number}
 * @private
 */
goog.testing.TestCase.prototype.startTime_ = 0;


/**
 * Time since the last batch of tests was started, if batchTime exceeds
 * {@link #maxRunTime} a timeout will be used to stop the tests blocking the
 * browser and a new batch will be started.
 * @type {number}
 * @private
 */
goog.testing.TestCase.prototype.batchTime_ = 0;


/**
 * Pointer to the current test.
 * @type {number}
 * @private
 */
goog.testing.TestCase.prototype.currentTestPointer_ = 0;


/**
 * Optional callback that will be executed when the test has finalized.
 * @type {Function}
 * @private
 */
goog.testing.TestCase.prototype.onCompleteCallback_ = null;


/**
 * Adds a new test to the test case.
 * @param {goog.testing.TestCase.Test} test The test to add.
 */
goog.testing.TestCase.prototype.add = function(test) {
  if (this.started) {
    throw Error('Tests cannot be added after execute() has been called. ' +
                'Test: ' + test.name);
  }

  this.tests_.push(test);
};


/**
 * Creates and adds a new test.
 *
 * Convenience function to make syntax less awkward when not using automatic
 * test discovery.
 *
 * @param {string} name The test name.
 * @param {!Function} ref Reference to the test function.
 * @param {!Object=} opt_scope Optional scope that the test function should be
 *     called in.
 */
goog.testing.TestCase.prototype.addNewTest = function(name, ref, opt_scope) {
  var test = new goog.testing.TestCase.Test(name, ref, opt_scope || this);
  this.add(test);
};


/**
 * Sets the tests.
 * @param {!Array.<goog.testing.TestCase.Test>} tests A new test array.
 * @protected
 */
goog.testing.TestCase.prototype.setTests = function(tests) {
  this.tests_ = tests;
};


/**
 * Gets the tests.
 * @return {!Array.<goog.testing.TestCase.Test>} The test array.
 */
goog.testing.TestCase.prototype.getTests = function() {
  return this.tests_;
};


/**
 * Returns the number of tests contained in the test case.
 * @return {number} The number of tests.
 */
goog.testing.TestCase.prototype.getCount = function() {
  return this.tests_.length;
};


/**
 * Returns the number of tests actually run in the test case, i.e. subtracting
 * any which are skipped.
 * @return {number} The number of un-ignored tests.
 */
goog.testing.TestCase.prototype.getActuallyRunCount = function() {
  return this.testsToRun_ ? goog.object.getCount(this.testsToRun_) : 0;
};


/**
 * Returns the current test and increments the pointer.
 * @return {goog.testing.TestCase.Test} The current test case.
 */
goog.testing.TestCase.prototype.next = function() {
  var test;
  while ((test = this.tests_[this.currentTestPointer_++])) {
    if (!this.testsToRun_ || this.testsToRun_[test.name] ||
        this.testsToRun_[this.currentTestPointer_ - 1]) {
      return test;
    }
  }
  return null;
};


/**
 * Resets the test case pointer, so that next returns the first test.
 */
goog.testing.TestCase.prototype.reset = function() {
  this.currentTestPointer_ = 0;
  this.result_ = new goog.testing.TestCase.Result(this);
};


/**
 * Sets the callback function that should be executed when the tests have
 * completed.
 * @param {Function} fn The callback function.
 */
goog.testing.TestCase.prototype.setCompletedCallback = function(fn) {
  this.onCompleteCallback_ = fn;
};


/**
 * Can be overridden in test classes to indicate whether the tests in a case
 * should be run in that particular situation.  For example, this could be used
 * to stop tests running in a particular browser, where browser support for
 * the class under test was absent.
 * @return {boolean} Whether any of the tests in the case should be run.
 */
goog.testing.TestCase.prototype.shouldRunTests = function() {
  return true;
};


/**
 * Executes each of the tests.
 */
goog.testing.TestCase.prototype.execute = function() {
  this.started = true;
  this.reset();
  this.startTime_ = this.now();
  this.running = true;
  this.result_.totalCount = this.getCount();

  if (!this.shouldRunTests()) {
    this.log('shouldRunTests() returned false, skipping these tests.');
    this.result_.testSuppressed = true;
    this.finalize();
    return;
  }

  this.log('Starting tests: ' + this.name_);
  this.cycleTests();
};


/**
 * Finalizes the test case, called when the tests have finished executing.
 */
goog.testing.TestCase.prototype.finalize = function() {
  this.saveMessage('Done');

  this.tearDownPage();

  var restoredSetTimeout =
      goog.testing.TestCase.protectedSetTimeout_ == goog.global.setTimeout &&
      goog.testing.TestCase.protectedClearTimeout_ == goog.global.clearTimeout;
  if (!restoredSetTimeout && goog.testing.TestCase.IS_IE &&
      String(goog.global.setTimeout) ==
          goog.testing.TestCase.setTimeoutAsString_) {
    // In strange cases, IE's value of setTimeout *appears* to change, but
    // the string representation stays stable.
    restoredSetTimeout = true;
  }

  if (!restoredSetTimeout) {
    var message = 'ERROR: Test did not restore setTimeout and clearTimeout';
    this.saveMessage(message);
    var err = new goog.testing.TestCase.Error(this.name_, message);
    this.result_.errors.push(err);
  }
  goog.global.clearTimeout = goog.testing.TestCase.protectedClearTimeout_;
  goog.global.setTimeout = goog.testing.TestCase.protectedSetTimeout_;
  this.endTime_ = this.now();
  this.running = false;
  this.result_.runTime = this.endTime_ - this.startTime_;
  this.result_.numFilesLoaded = this.countNumFilesLoaded_();
  this.result_.complete = true;

  this.log(this.result_.getSummary());
  if (this.result_.isSuccess()) {
    this.log('Tests complete');
  } else {
    this.log('Tests Failed');
  }
  if (this.onCompleteCallback_) {
    var fn = this.onCompleteCallback_;
    // Execute's the completed callback in the context of the global object.
    fn();
    this.onCompleteCallback_ = null;
  }
};


/**
 * Saves a message to the result set.
 * @param {string} message The message to save.
 */
goog.testing.TestCase.prototype.saveMessage = function(message) {
  this.result_.messages.push(this.getTimeStamp_() + '  ' + message);
};


/**
 * @return {boolean} Whether the test case is running inside the multi test
 *     runner.
 */
goog.testing.TestCase.prototype.isInsideMultiTestRunner = function() {
  var top = goog.global['top'];
  return top && typeof top['_allTests'] != 'undefined';
};


/**
 * Logs an object to the console, if available.
 * @param {*} val The value to log. Will be ToString'd.
 */
goog.testing.TestCase.prototype.log = function(val) {
  if (!this.isInsideMultiTestRunner() && goog.global.console) {
    if (typeof val == 'string') {
      val = this.getTimeStamp_() + ' : ' + val;
    }
    if (val instanceof Error && val.stack) {
      // Chrome does console.log asynchronously in a different process
      // (http://code.google.com/p/chromium/issues/detail?id=50316).
      // This is an acute problem for Errors, which almost never survive.
      // Grab references to the immutable strings so they survive.
      goog.global.console.log(val, val.message, val.stack);
      // TODO(gboyer): Consider for Chrome cloning any object if we can ensure
      // there are no circular references.
    } else {
      goog.global.console.log(val);
    }
  }
};


/**
 * @return {boolean} Whether the test was a success.
 */
goog.testing.TestCase.prototype.isSuccess = function() {
  return !!this.result_ && this.result_.isSuccess();
};


/**
 * Returns a string detailing the results from the test.
 * @param {boolean=} opt_verbose If true results will include data about all
 *     tests, not just what failed.
 * @return {string} The results from the test.
 */
goog.testing.TestCase.prototype.getReport = function(opt_verbose) {
  var rv = [];

  if (this.running) {
    rv.push(this.name_ + ' [RUNNING]');
  } else {
    var label = this.result_.isSuccess() ? 'PASSED' : 'FAILED';
    rv.push(this.name_ + ' [' + label + ']');
  }

  if (goog.global.location) {
    rv.push(this.trimPath_(goog.global.location.href));
  }

  rv.push(this.result_.getSummary());

  if (opt_verbose) {
    rv.push('.', this.result_.messages.join('\n'));
  } else if (!this.result_.isSuccess()) {
    rv.push(this.result_.errors.join('\n'));
  }

  rv.push(' ');

  return rv.join('\n');
};


/**
 * Returns the amount of time it took for the test to run.
 * @return {number} The run time, in milliseconds.
 */
goog.testing.TestCase.prototype.getRunTime = function() {
  return this.result_.runTime;
};


/**
 * Returns the number of script files that were loaded in order to run the test.
 * @return {number} The number of script files.
 */
goog.testing.TestCase.prototype.getNumFilesLoaded = function() {
  return this.result_.numFilesLoaded;
};


/**
 * Returns the test results object: a map from test names to a list of test
 * failures (if any exist).
 * @return {!Object.<string, !Array.<string>>} Tests results object.
 */
goog.testing.TestCase.prototype.getTestResults = function() {
  return this.result_.resultsByName;
};


/**
 * Executes each of the tests.
 * Overridable by the individual test case.  This allows test cases to defer
 * when the test is actually started.  If overridden, finalize must be called
 * by the test to indicate it has finished.
 */
goog.testing.TestCase.prototype.runTests = function() {
  try {
    this.setUpPage();
  } catch (e) {
    this.exceptionBeforeTest = e;
  }
  this.execute();
};


/**
 * Reorders the tests depending on the {@code order} field.
 * @param {Array.<goog.testing.TestCase.Test>} tests An array of tests to
 *     reorder.
 * @private
 */
goog.testing.TestCase.prototype.orderTests_ = function(tests) {
  switch (this.order) {
    case goog.testing.TestCase.Order.RANDOM:
      // Fisher-Yates shuffle
      var i = tests.length;
      while (i > 1) {
        // goog.math.randomInt is inlined to reduce dependencies.
        var j = Math.floor(Math.random() * i); // exclusive
        i--;
        var tmp = tests[i];
        tests[i] = tests[j];
        tests[j] = tmp;
      }
      break;

    case goog.testing.TestCase.Order.SORTED:
      tests.sort(function(t1, t2) {
        if (t1.name == t2.name) {
          return 0;
        }
        return t1.name < t2.name ? -1 : 1;
      });
      break;

      // Do nothing for NATURAL.
  }
};


/**
 * Gets list of objects that potentially contain test cases. For IE 8 and below,
 * this is the global "this" (for properties set directly on the global this or
 * window) and the RuntimeObject (for global variables and functions). For all
 * other browsers, the array simply contains the global this.
 *
 * @param {string=} opt_prefix An optional prefix. If specified, only get things
 *     under this prefix. Note that the prefix is only honored in IE, since it
 *     supports the RuntimeObject:
 *     http://msdn.microsoft.com/en-us/library/ff521039%28VS.85%29.aspx
 *     TODO: Remove this option.
 * @return {!Array.<!Object>} A list of objects that should be inspected.
 */
goog.testing.TestCase.prototype.getGlobals = function(opt_prefix) {
  return goog.testing.TestCase.getGlobals(opt_prefix);
};


/**
 * Gets list of objects that potentially contain test cases. For IE 8 and below,
 * this is the global "this" (for properties set directly on the global this or
 * window) and the RuntimeObject (for global variables and functions). For all
 * other browsers, the array simply contains the global this.
 *
 * @param {string=} opt_prefix An optional prefix. If specified, only get things
 *     under this prefix. Note that the prefix is only honored in IE, since it
 *     supports the RuntimeObject:
 *     http://msdn.microsoft.com/en-us/library/ff521039%28VS.85%29.aspx
 *     TODO: Remove this option.
 * @return {!Array.<!Object>} A list of objects that should be inspected.
 */
goog.testing.TestCase.getGlobals = function(opt_prefix) {
  // Look in the global scope for most browsers, on IE we use the little known
  // RuntimeObject which holds references to all globals. We reference this
  // via goog.global so that there isn't an aliasing that throws an exception
  // in Firefox.
  return typeof goog.global['RuntimeObject'] != 'undefined' ?
      [goog.global['RuntimeObject']((opt_prefix || '') + '*'), goog.global] :
      [goog.global];
};


/**
 * Gets called before any tests are executed.  Can be overridden to set up the
 * environment for the whole test case.
 */
goog.testing.TestCase.prototype.setUpPage = function() {};


/**
 * Gets called after all tests have been executed.  Can be overridden to tear
 * down the entire test case.
 */
goog.testing.TestCase.prototype.tearDownPage = function() {};


/**
 * Gets called before every goog.testing.TestCase.Test is been executed. Can be
 * overridden to add set up functionality to each test.
 */
goog.testing.TestCase.prototype.setUp = function() {};


/**
 * Gets called after every goog.testing.TestCase.Test has been executed. Can be
 * overriden to add tear down functionality to each test.
 */
goog.testing.TestCase.prototype.tearDown = function() {};


/**
 * @return {string} The function name prefix used to auto-discover tests.
 * @protected
 */
goog.testing.TestCase.prototype.getAutoDiscoveryPrefix = function() {
  return 'test';
};


/**
 * @return {number} Time since the last batch of tests was started.
 * @protected
 */
goog.testing.TestCase.prototype.getBatchTime = function() {
  return this.batchTime_;
};


/**
 * @param {number} batchTime Time since the last batch of tests was started.
 * @protected
 */
goog.testing.TestCase.prototype.setBatchTime = function(batchTime) {
  this.batchTime_ = batchTime;
};


/**
 * Creates a {@code goog.testing.TestCase.Test} from an auto-discovered
 *     function.
 * @param {string} name The name of the function.
 * @param {function() : void} ref The auto-discovered function.
 * @return {!goog.testing.TestCase.Test} The newly created test.
 * @protected
 */
goog.testing.TestCase.prototype.createTestFromAutoDiscoveredFunction =
    function(name, ref) {
  return new goog.testing.TestCase.Test(name, ref, goog.global);
};


/**
 * Adds any functions defined in the global scope that correspond to
 * lifecycle events for the test case. Overrides setUp, tearDown, setUpPage,
 * tearDownPage and runTests if they are defined.
 */
goog.testing.TestCase.prototype.autoDiscoverLifecycle = function() {
  if (goog.global['setUp']) {
    this.setUp = goog.bind(goog.global['setUp'], goog.global);
  }
  if (goog.global['tearDown']) {
    this.tearDown = goog.bind(goog.global['tearDown'], goog.global);
  }
  if (goog.global['setUpPage']) {
    this.setUpPage = goog.bind(goog.global['setUpPage'], goog.global);
  }
  if (goog.global['tearDownPage']) {
    this.tearDownPage = goog.bind(goog.global['tearDownPage'], goog.global);
  }
  if (goog.global['runTests']) {
    this.runTests = goog.bind(goog.global['runTests'], goog.global);
  }
  if (goog.global['shouldRunTests']) {
    this.shouldRunTests = goog.bind(goog.global['shouldRunTests'], goog.global);
  }
};


/**
 * Adds any functions defined in the global scope that are prefixed with "test"
 * to the test case.
 */
goog.testing.TestCase.prototype.autoDiscoverTests = function() {
  var prefix = this.getAutoDiscoveryPrefix();
  var testSources = this.getGlobals(prefix);

  var foundTests = [];

  for (var i = 0; i < testSources.length; i++) {
    var testSource = testSources[i];
    for (var name in testSource) {
      if ((new RegExp('^' + prefix)).test(name)) {
        var ref;
        try {
          ref = testSource[name];
        } catch (ex) {
          // NOTE(brenneman): When running tests from a file:// URL on Firefox
          // 3.5 for Windows, any reference to goog.global.sessionStorage raises
          // an "Operation is not supported" exception. Ignore any exceptions
          // raised by simply accessing global properties.
          ref = undefined;
        }

        if (goog.isFunction(ref)) {
          foundTests.push(this.createTestFromAutoDiscoveredFunction(name, ref));
        }
      }
    }
  }

  this.orderTests_(foundTests);

  for (var i = 0; i < foundTests.length; i++) {
    this.add(foundTests[i]);
  }

  this.log(this.getCount() + ' tests auto-discovered');

  // TODO(user): Do this as a separate call. Unfortunately, a lot of projects
  // currently override autoDiscoverTests and expect lifecycle events to be
  // registered as a part of this call.
  this.autoDiscoverLifecycle();
};


/**
 * Checks to see if the test should be marked as failed before it is run.
 *
 * If there was an error in setUpPage, we treat that as a failure for all tests
 * and mark them all as having failed.
 *
 * @param {goog.testing.TestCase.Test} testCase The current test case.
 * @return {boolean} Whether the test was marked as failed.
 * @protected
 */
goog.testing.TestCase.prototype.maybeFailTestEarly = function(testCase) {
  if (this.exceptionBeforeTest) {
    // We just use the first error to report an error on a failed test.
    testCase.name = 'setUpPage for ' + testCase.name;
    this.doError(testCase, this.exceptionBeforeTest);
    return true;
  }
  return false;
};


/**
 * Cycles through the tests, breaking out using a setTimeout if the execution
 * time has execeeded {@link #maxRunTime}.
 */
goog.testing.TestCase.prototype.cycleTests = function() {
  this.saveMessage('Start');
  this.batchTime_ = this.now();
  var nextTest;
  while ((nextTest = this.next()) && this.running) {
    this.result_.runCount++;
    // Execute the test and handle the error, we execute all tests rather than
    // stopping after a single error.
    var cleanedUp = false;
    try {
      this.log('Running test: ' + nextTest.name);

      if (this.maybeFailTestEarly(nextTest)) {
        cleanedUp = true;
      } else {
        goog.testing.TestCase.currentTestName = nextTest.name;
        this.setUp();
        nextTest.execute();
        this.tearDown();
        goog.testing.TestCase.currentTestName = null;

        cleanedUp = true;

        this.doSuccess(nextTest);
      }
    } catch (e) {
      this.doError(nextTest, e);

      if (!cleanedUp) {
        try {
          this.tearDown();
        } catch (e2) {} // Fail silently if tearDown is throwing the errors.
      }
    }

    // If the max run time is exceeded call this function again async so as not
    // to block the browser.
    if (this.currentTestPointer_ < this.tests_.length &&
        this.now() - this.batchTime_ > goog.testing.TestCase.maxRunTime) {
      this.saveMessage('Breaking async');
      this.timeout(goog.bind(this.cycleTests, this), 0);
      return;
    }
  }
  // Tests are done.
  this.finalize();
};


/**
 * Counts the number of files that were loaded for dependencies that are
 * required to run the test.
 * @return {number} The number of files loaded.
 * @private
 */
goog.testing.TestCase.prototype.countNumFilesLoaded_ = function() {
  var scripts = document.getElementsByTagName('script');
  var count = 0;
  for (var i = 0, n = scripts.length; i < n; i++) {
    if (scripts[i].src) {
      count++;
    }
  }
  return count;
};


/**
 * Calls a function after a delay, using the protected timeout.
 * @param {Function} fn The function to call.
 * @param {number} time Delay in milliseconds.
 * @return {number} The timeout id.
 * @protected
 */
goog.testing.TestCase.prototype.timeout = function(fn, time) {
  // NOTE: invoking protectedSetTimeout_ as a member of goog.testing.TestCase
  // would result in an Illegal Invocation error. The method must be executed
  // with the global context.
  var protectedSetTimeout = goog.testing.TestCase.protectedSetTimeout_;
  return protectedSetTimeout(fn, time);
};


/**
 * Clears a timeout created by {@code this.timeout()}.
 * @param {number} id A timeout id.
 * @protected
 */
goog.testing.TestCase.prototype.clearTimeout = function(id) {
  // NOTE: see execution note for protectedSetTimeout above.
  var protectedClearTimeout = goog.testing.TestCase.protectedClearTimeout_;
  protectedClearTimeout(id);
};


/**
 * @return {number} The current time in milliseconds, don't use goog.now as some
 *     tests override it.
 * @protected
 */
goog.testing.TestCase.prototype.now = function() {
  // Cannot use "new goog.testing.TestCase.protectedDate_()" due to b/8323223.
  var protectedDate = goog.testing.TestCase.protectedDate_;
  return new protectedDate().getTime();
};


/**
 * Returns the current time.
 * @return {string} HH:MM:SS.
 * @private
 */
goog.testing.TestCase.prototype.getTimeStamp_ = function() {
  // Cannot use "new goog.testing.TestCase.protectedDate_()" due to b/8323223.
  var protectedDate = goog.testing.TestCase.protectedDate_;
  var d = new protectedDate();

  // Ensure millis are always 3-digits
  var millis = '00' + d.getMilliseconds();
  millis = millis.substr(millis.length - 3);

  return this.pad_(d.getHours()) + ':' + this.pad_(d.getMinutes()) + ':' +
         this.pad_(d.getSeconds()) + '.' + millis;
};


/**
 * Pads a number to make it have a leading zero if it's less than 10.
 * @param {number} number The number to pad.
 * @return {string} The resulting string.
 * @private
 */
goog.testing.TestCase.prototype.pad_ = function(number) {
  return number < 10 ? '0' + number : String(number);
};


/**
 * Trims a path to be only that after google3.
 * @param {string} path The path to trim.
 * @return {string} The resulting string.
 * @private
 */
goog.testing.TestCase.prototype.trimPath_ = function(path) {
  return path.substring(path.indexOf('google3') + 8);
};


/**
 * Handles a test that passed.
 * @param {goog.testing.TestCase.Test} test The test that passed.
 * @protected
 */
goog.testing.TestCase.prototype.doSuccess = function(test) {
  this.result_.successCount++;
  // An empty list of error messages indicates that the test passed.
  // If we already have a failure for this test, do not set to empty list.
  if (!(test.name in this.result_.resultsByName)) {
    this.result_.resultsByName[test.name] = [];
  }
  var message = test.name + ' : PASSED';
  this.saveMessage(message);
  this.log(message);
};


/**
 * Handles a test that failed.
 * @param {goog.testing.TestCase.Test} test The test that failed.
 * @param {*=} opt_e The exception object associated with the
 *     failure or a string.
 * @protected
 */
goog.testing.TestCase.prototype.doError = function(test, opt_e) {
  var message = test.name + ' : FAILED';
  this.log(message);
  this.saveMessage(message);
  var err = this.logError(test.name, opt_e);
  this.result_.errors.push(err);
  if (test.name in this.result_.resultsByName) {
    this.result_.resultsByName[test.name].push(err.toString());
  } else {
    this.result_.resultsByName[test.name] = [err.toString()];
  }
};


/**
 * @param {string} name Failed test name.
 * @param {*=} opt_e The exception object associated with the
 *     failure or a string.
 * @return {!goog.testing.TestCase.Error} Error object.
 */
goog.testing.TestCase.prototype.logError = function(name, opt_e) {
  var errMsg = null;
  var stack = null;
  if (opt_e) {
    this.log(opt_e);
    if (goog.isString(opt_e)) {
      errMsg = opt_e;
    } else {
      errMsg = opt_e.message || opt_e.description || opt_e.toString();
      stack = opt_e.stack ? goog.testing.stacktrace.canonicalize(opt_e.stack) :
          opt_e['stackTrace'];
    }
  } else {
    errMsg = 'An unknown error occurred';
  }
  var err = new goog.testing.TestCase.Error(name, errMsg, stack);

  // Avoid double logging.
  if (!opt_e || !opt_e['isJsUnitException'] ||
      !opt_e['loggedJsUnitException']) {
    this.saveMessage(err.toString());
  }
  if (opt_e && opt_e['isJsUnitException']) {
    opt_e['loggedJsUnitException'] = true;
  }

  return err;
};



/**
 * A class representing a single test function.
 * @param {string} name The test name.
 * @param {Function} ref Reference to the test function.
 * @param {Object=} opt_scope Optional scope that the test function should be
 *     called in.
 * @constructor
 */
goog.testing.TestCase.Test = function(name, ref, opt_scope) {
  /**
   * The name of the test.
   * @type {string}
   */
  this.name = name;

  /**
   * Reference to the test function.
   * @type {Function}
   */
  this.ref = ref;

  /**
   * Scope that the test function should be called in.
   * @type {Object}
   */
  this.scope = opt_scope || null;
};


/**
 * Executes the test function.
 */
goog.testing.TestCase.Test.prototype.execute = function() {
  this.ref.call(this.scope);
};



/**
 * A class for representing test results.  A bag of public properties.
 * @param {goog.testing.TestCase} testCase The test case that owns this result.
 * @constructor
 * @final
 */
goog.testing.TestCase.Result = function(testCase) {
  /**
   * The test case that owns this result.
   * @type {goog.testing.TestCase}
   * @private
   */
  this.testCase_ = testCase;

  /**
   * Total number of tests that should have been run.
   * @type {number}
   */
  this.totalCount = 0;

  /**
   * Total number of tests that were actually run.
   * @type {number}
   */
  this.runCount = 0;

  /**
   * Number of successful tests.
   * @type {number}
   */
  this.successCount = 0;

  /**
   * The amount of time the tests took to run.
   * @type {number}
   */
  this.runTime = 0;

  /**
   * The number of files loaded to run this test.
   * @type {number}
   */
  this.numFilesLoaded = 0;

  /**
   * Whether this test case was suppressed by shouldRunTests() returning false.
   * @type {boolean}
   */
  this.testSuppressed = false;

  /**
   * Test results for each test that was run. The test name is always added
   * as the key in the map, and the array of strings is an optional list
   * of failure messages. If the array is empty, the test passed. Otherwise,
   * the test failed.
   * @type {!Object.<string, !Array.<string>>}
   */
  this.resultsByName = {};

  /**
   * Errors encountered while running the test.
   * @type {!Array.<goog.testing.TestCase.Error>}
   */
  this.errors = [];

  /**
   * Messages to show the user after running the test.
   * @type {!Array.<string>}
   */
  this.messages = [];

  /**
   * Whether the tests have completed.
   * @type {boolean}
   */
  this.complete = false;
};


/**
 * @return {boolean} Whether the test was successful.
 */
goog.testing.TestCase.Result.prototype.isSuccess = function() {
  return this.complete && this.errors.length == 0;
};


/**
 * @return {string} A summary of the tests, including total number of tests that
 *     passed, failed, and the time taken.
 */
goog.testing.TestCase.Result.prototype.getSummary = function() {
  var summary = this.runCount + ' of ' + this.totalCount + ' tests run in ' +
      this.runTime + 'ms.\n';
  if (this.testSuppressed) {
    summary += 'Tests not run because shouldRunTests() returned false.';
  } else {
    var failures = this.totalCount - this.successCount;
    var suppressionMessage = '';

    var countOfRunTests = this.testCase_.getActuallyRunCount();
    if (countOfRunTests) {
      failures = countOfRunTests - this.successCount;
      suppressionMessage = ', ' +
          (this.totalCount - countOfRunTests) + ' suppressed by querystring';
    }
    summary += this.successCount + ' passed, ' +
        failures + ' failed' + suppressionMessage + '.\n' +
        Math.round(this.runTime / this.runCount) + ' ms/test. ' +
        this.numFilesLoaded + ' files loaded.';
  }

  return summary;
};


/**
 * Initializes the given test case with the global test runner 'G_testRunner'.
 * @param {goog.testing.TestCase} testCase The test case to install.
 */
goog.testing.TestCase.initializeTestRunner = function(testCase) {
  testCase.autoDiscoverTests();
  var gTestRunner = goog.global['G_testRunner'];
  if (gTestRunner) {
    gTestRunner['initialize'](testCase);
  } else {
    throw Error('G_testRunner is undefined. Please ensure goog.testing.jsunit' +
        ' is included.');
  }
};



/**
 * A class representing an error thrown by the test
 * @param {string} source The name of the test which threw the error.
 * @param {string} message The error message.
 * @param {string=} opt_stack A string showing the execution stack.
 * @constructor
 * @final
 */
goog.testing.TestCase.Error = function(source, message, opt_stack) {
  /**
   * The name of the test which threw the error.
   * @type {string}
   */
  this.source = source;

  /**
   * Reference to the test function.
   * @type {string}
   */
  this.message = message;

  /**
   * Scope that the test function should be called in.
   * @type {?string}
   */
  this.stack = opt_stack || null;
};


/**
 * Returns a string representing the error object.
 * @return {string} A string representation of the error.
 * @override
 */
goog.testing.TestCase.Error.prototype.toString = function() {
  return 'ERROR in ' + this.source + '\n' +
      this.message + (this.stack ? '\n' + this.stack : '');
};
