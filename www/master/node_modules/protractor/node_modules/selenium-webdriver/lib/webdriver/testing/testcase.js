// Copyright 2011 Software Freedom Conservancy. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Defines a special test case that runs each test inside of a
 * {@code webdriver.Application}. This allows each phase to schedule
 * asynchronous actions that run to completion before the next phase of the
 * test.
 *
 * This file requires the global {@code G_testRunner} to be initialized before
 * use. This can be accomplished by also importing
 * {@link webdriver.testing.jsunit}. This namespace is not required by default
 * to improve interoperability with other namespaces that may initialize
 * G_testRunner.
 */

goog.provide('webdriver.testing.TestCase');

goog.require('goog.testing.TestCase');
goog.require('webdriver.promise.ControlFlow');
/** @suppress {extraRequire} Imported for user convenience. */
goog.require('webdriver.testing.asserts');



/**
 * Constructs a test case that synchronizes each test case with the singleton
 * {@code webdriver.promise.ControlFlow}.
 *
 * @param {!webdriver.testing.Client} client The test client to use for
 *     reporting test results.
 * @param {string=} opt_name The name of the test case, defaults to
 *     'Untitled Test Case'.
 * @constructor
 * @extends {goog.testing.TestCase}
 */
webdriver.testing.TestCase = function(client, opt_name) {
  goog.base(this, opt_name);

  /** @private {!webdriver.testing.Client} */
  this.client_ = client;
};
goog.inherits(webdriver.testing.TestCase, goog.testing.TestCase);


/**
 * Executes the next test inside its own {@code webdriver.Application}.
 * @override
 */
webdriver.testing.TestCase.prototype.cycleTests = function() {
  var test = this.next();
  if (!test) {
    this.finalize();
    return;
  }

  goog.testing.TestCase.currentTestName = test.name;
  this.result_.runCount++;
  this.log('Running test: ' + test.name);
  this.client_.sendTestStartedEvent(test.name);

  var self = this;
  var hadError = false;
  var app = webdriver.promise.controlFlow();

  this.runSingleTest_(test, onError).then(function() {
    hadError || self.doSuccess(test);
    self.timeout(function() {
      self.cycleTests();
    }, 100);
  });

  function onError(e) {
    hadError = true;
    self.doError(test, app.annotateError(e));
    // Note: result_ is a @protected field but still uses the trailing
    // underscore.
    var err = self.result_.errors[self.result_.errors.length - 1];
    self.client_.sendErrorEvent(err.toString());
  }
};


/** @override */
webdriver.testing.TestCase.prototype.logError = function(name, opt_e) {
  var errMsg = null;
  var stack = null;
  if (opt_e) {
    this.log(opt_e);
    if (goog.isString(opt_e)) {
      errMsg = opt_e;
    } else {
      // In case someone calls this function directly, make sure we have a
      // properly annotated error.
      webdriver.promise.controlFlow().annotateError(opt_e);
      errMsg = opt_e.toString();
      stack = opt_e.stack.substring(errMsg.length + 1);
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
 * Executes a single test, scheduling each phase with the global application.
 * Each phase will wait for the application to go idle before moving on to the
 * next test phase.  This function models the follow basic test flow:
 *
 *   try {
 *     this.setUp.call(test.scope);
 *     test.ref.call(test.scope);
 *   } catch (ex) {
 *     onError(ex);
 *   } finally {
 *     try {
 *       this.tearDown.call(test.scope);
 *     } catch (e) {
 *       onError(e);
 *     }
 *   }
 *
 * @param {!goog.testing.TestCase.Test} test The test to run.
 * @param {function(*)} onError The function to call each time an error is
 *     detected.
 * @return {!webdriver.promise.Promise} A promise that will be resolved when the
 *     test has finished running.
 * @private
 */
webdriver.testing.TestCase.prototype.runSingleTest_ = function(test, onError) {
  var flow = webdriver.promise.controlFlow();
  flow.clearHistory();

  return execute(test.name + '.setUp()', this.setUp)().
      then(execute(test.name + '()', test.ref)).
      thenCatch(onError).
      then(execute(test.name + '.tearDown()', this.tearDown)).
      thenCatch(onError);

  function execute(description, fn) {
    return function() {
      return flow.execute(goog.bind(fn, test.scope), description);
    }
  }
};
