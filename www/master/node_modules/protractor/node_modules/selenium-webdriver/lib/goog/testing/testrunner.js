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
 * @fileoverview The test runner is a singleton object that is used to execute
 * a goog.testing.TestCases, display the results, and expose the results to
 * Selenium for automation.  If a TestCase hasn't been registered with the
 * runner by the time window.onload occurs, the testRunner will try to auto-
 * discover JsUnit style test pages.
 *
 * The hooks for selenium are (see http://go/selenium-hook-setup):-
 *  - Boolean G_testRunner.isFinished()
 *  - Boolean G_testRunner.isSuccess()
 *  - String G_testRunner.getReport()
 *  - number G_testRunner.getRunTime()
 *  - Object.<string, Array.<string>> G_testRunner.getTestResults()
 *
 * Testing code should not have dependencies outside of goog.testing so as to
 * reduce the chance of masking missing dependencies.
 *
 */

goog.provide('goog.testing.TestRunner');

goog.require('goog.testing.TestCase');



/**
 * Construct a test runner.
 *
 * NOTE(user): This is currently pretty weird, I'm essentially trying to
 * create a wrapper that the Selenium test can hook into to query the state of
 * the running test case, while making goog.testing.TestCase general.
 *
 * @constructor
 */
goog.testing.TestRunner = function() {
  /**
   * Errors that occurred in the window.
   * @type {Array.<string>}
   */
  this.errors = [];
};


/**
 * Reference to the active test case.
 * @type {goog.testing.TestCase?}
 */
goog.testing.TestRunner.prototype.testCase = null;


/**
 * Whether the test runner has been initialized yet.
 * @type {boolean}
 */
goog.testing.TestRunner.prototype.initialized = false;


/**
 * Element created in the document to add test results to.
 * @type {Element}
 * @private
 */
goog.testing.TestRunner.prototype.logEl_ = null;


/**
 * Function to use when filtering errors.
 * @type {(function(string))?}
 * @private
 */
goog.testing.TestRunner.prototype.errorFilter_ = null;


/**
 * Whether an empty test case counts as an error.
 * @type {boolean}
 * @private
 */
goog.testing.TestRunner.prototype.strict_ = true;


/**
 * Initializes the test runner.
 * @param {goog.testing.TestCase} testCase The test case to initialize with.
 */
goog.testing.TestRunner.prototype.initialize = function(testCase) {
  if (this.testCase && this.testCase.running) {
    throw Error('The test runner is already waiting for a test to complete');
  }
  this.testCase = testCase;
  this.initialized = true;
};


/**
 * By default, the test runner is strict, and fails if it runs an empty
 * test case.
 * @param {boolean} strict Whether the test runner should fail on an empty
 *     test case.
 */
goog.testing.TestRunner.prototype.setStrict = function(strict) {
  this.strict_ = strict;
};


/**
 * @return {boolean} Whether the test runner should fail on an empty
 *     test case.
 */
goog.testing.TestRunner.prototype.isStrict = function() {
  return this.strict_;
};


/**
 * Returns true if the test runner is initialized.
 * Used by Selenium Hooks.
 * @return {boolean} Whether the test runner is active.
 */
goog.testing.TestRunner.prototype.isInitialized = function() {
  return this.initialized;
};


/**
 * Returns true if the test runner is finished.
 * Used by Selenium Hooks.
 * @return {boolean} Whether the test runner is active.
 */
goog.testing.TestRunner.prototype.isFinished = function() {
  return this.errors.length > 0 ||
      this.initialized && !!this.testCase && this.testCase.started &&
      !this.testCase.running;
};


/**
 * Returns true if the test case didn't fail.
 * Used by Selenium Hooks.
 * @return {boolean} Whether the current test returned successfully.
 */
goog.testing.TestRunner.prototype.isSuccess = function() {
  return !this.hasErrors() && !!this.testCase && this.testCase.isSuccess();
};


/**
 * Returns true if the test case runner has errors that were caught outside of
 * the test case.
 * @return {boolean} Whether there were JS errors.
 */
goog.testing.TestRunner.prototype.hasErrors = function() {
  return this.errors.length > 0;
};


/**
 * Logs an error that occurred.  Used in the case of environment setting up
 * an onerror handler.
 * @param {string} msg Error message.
 */
goog.testing.TestRunner.prototype.logError = function(msg) {
  if (!this.errorFilter_ || this.errorFilter_.call(null, msg)) {
    this.errors.push(msg);
  }
};


/**
 * Log failure in current running test.
 * @param {Error} ex Exception.
 */
goog.testing.TestRunner.prototype.logTestFailure = function(ex) {
  var testName = /** @type {string} */ (goog.testing.TestCase.currentTestName);
  if (this.testCase) {
    this.testCase.logError(testName, ex);
  } else {
    // NOTE: Do not forget to log the original exception raised.
    throw new Error('Test runner not initialized with a test case. Original ' +
                    'exception: ' + ex.message);
  }
};


/**
 * Sets a function to use as a filter for errors.
 * @param {function(string)} fn Filter function.
 */
goog.testing.TestRunner.prototype.setErrorFilter = function(fn) {
  this.errorFilter_ = fn;
};


/**
 * Returns a report of the test case that ran.
 * Used by Selenium Hooks.
 * @param {boolean=} opt_verbose If true results will include data about all
 *     tests, not just what failed.
 * @return {string} A report summary of the test.
 */
goog.testing.TestRunner.prototype.getReport = function(opt_verbose) {
  var report = [];
  if (this.testCase) {
    report.push(this.testCase.getReport(opt_verbose));
  }
  if (this.errors.length > 0) {
    report.push('JavaScript errors detected by test runner:');
    report.push.apply(report, this.errors);
    report.push('\n');
  }
  return report.join('\n');
};


/**
 * Returns the amount of time it took for the test to run.
 * Used by Selenium Hooks.
 * @return {number} The run time, in milliseconds.
 */
goog.testing.TestRunner.prototype.getRunTime = function() {
  return this.testCase ? this.testCase.getRunTime() : 0;
};


/**
 * Returns the number of script files that were loaded in order to run the test.
 * @return {number} The number of script files.
 */
goog.testing.TestRunner.prototype.getNumFilesLoaded = function() {
  return this.testCase ? this.testCase.getNumFilesLoaded() : 0;
};


/**
 * Executes a test case and prints the results to the window.
 */
goog.testing.TestRunner.prototype.execute = function() {
  if (!this.testCase) {
    throw Error('The test runner must be initialized with a test case ' +
                'before execute can be called.');
  }

  if (this.strict_ && this.testCase.getCount() == 0) {
    throw Error(
        'No tests found in given test case: ' +
        this.testCase.getName() + ' ' +
        'By default, the test runner fails if a test case has no tests. ' +
        'To modify this behavior, see goog.testing.TestRunner\'s ' +
        'setStrict() method, or G_testRunner.setStrict()');
  }

  this.testCase.setCompletedCallback(goog.bind(this.onComplete_, this));
  this.testCase.runTests();
};


/**
 * Writes the results to the document when the test case completes.
 * @private
 */
goog.testing.TestRunner.prototype.onComplete_ = function() {
  var log = this.testCase.getReport(true);
  if (this.errors.length > 0) {
    log += '\n' + this.errors.join('\n');
  }

  if (!this.logEl_) {
    var el = document.getElementById('closureTestRunnerLog');
    if (el == null) {
      el = document.createElement('div');
      document.body.appendChild(el);
    }
    this.logEl_ = el;
  }

  // Highlight the page to indicate the overall outcome.
  this.writeLog(log);

  // TODO(user): Make this work with multiple test cases (b/8603638).
  var runAgainLink = document.createElement('a');
  runAgainLink.style.display = 'inline-block';
  runAgainLink.style.fontSize = 'small';
  runAgainLink.style.marginBottom = '16px';
  runAgainLink.href = '';
  runAgainLink.onclick = goog.bind(function() {
    this.execute();
    return false;
  }, this);
  runAgainLink.innerHTML = 'Run again without reloading';
  this.logEl_.appendChild(runAgainLink);
};


/**
 * Writes a nicely formatted log out to the document.
 * @param {string} log The string to write.
 */
goog.testing.TestRunner.prototype.writeLog = function(log) {
  var lines = log.split('\n');
  for (var i = 0; i < lines.length; i++) {
    var line = lines[i];
    var color;
    var isFailOrError = /FAILED/.test(line) || /ERROR/.test(line);
    if (/PASSED/.test(line)) {
      color = 'darkgreen';
    } else if (isFailOrError) {
      color = 'darkred';
    } else {
      color = '#333';
    }
    var div = document.createElement('div');
    if (line.substr(0, 2) == '> ') {
      // The stack trace may contain links so it has to be interpreted as HTML.
      div.innerHTML = line;
    } else {
      div.appendChild(document.createTextNode(line));
    }

    var testNameMatch =
        /(\S+) (\[[^\]]*] )?: (FAILED|ERROR|PASSED)/.exec(line);
    if (testNameMatch) {
      // Build a URL to run the test individually.  If this test was already
      // part of another subset test, we need to overwrite the old runTests
      // query parameter.  We also need to do this without bringing in any
      // extra dependencies, otherwise we could mask missing dependency bugs.
      var newSearch = 'runTests=' + testNameMatch[1];
      var search = window.location.search;
      if (search) {
        var oldTests = /runTests=([^&]*)/.exec(search);
        if (oldTests) {
          newSearch = search.substr(0, oldTests.index) +
                      newSearch +
                      search.substr(oldTests.index + oldTests[0].length);
        } else {
          newSearch = search + '&' + newSearch;
        }
      } else {
        newSearch = '?' + newSearch;
      }
      var href = window.location.href;
      var hash = window.location.hash;
      if (hash && hash.charAt(0) != '#') {
        hash = '#' + hash;
      }
      href = href.split('#')[0].split('?')[0] + newSearch + hash;

      // Add the link.
      var a = document.createElement('A');
      a.innerHTML = '(run individually)';
      a.style.fontSize = '0.8em';
      a.style.color = '#888';
      a.href = href;
      div.appendChild(document.createTextNode(' '));
      div.appendChild(a);
    }

    div.style.color = color;
    div.style.font = 'normal 100% monospace';
    div.style.wordWrap = 'break-word';
    if (i == 0) {
      // Highlight the first line as a header that indicates the test outcome.
      div.style.padding = '20px';
      div.style.marginBottom = '10px';
      if (isFailOrError) {
        div.style.border = '5px solid ' + color;
        div.style.backgroundColor = '#ffeeee';
      } else {
        div.style.border = '1px solid black';
        div.style.backgroundColor = '#eeffee';
      }
    }

    try {
      div.style.whiteSpace = 'pre-wrap';
    } catch (e) {
      // NOTE(brenneman): IE raises an exception when assigning to pre-wrap.
      // Thankfully, it doesn't collapse whitespace when using monospace fonts,
      // so it will display correctly if we ignore the exception.
    }

    if (i < 2) {
      div.style.fontWeight = 'bold';
    }
    this.logEl_.appendChild(div);
  }
};


/**
 * Logs a message to the current test case.
 * @param {string} s The text to output to the log.
 */
goog.testing.TestRunner.prototype.log = function(s) {
  if (this.testCase) {
    this.testCase.log(s);
  }
};


// TODO(nnaze): Properly handle serving test results when multiple test cases
// are run.
/**
 * @return {Object.<string, !Array.<string>>} A map of test names to a list of
 * test failures (if any) to provide formatted data for the test runner.
 */
goog.testing.TestRunner.prototype.getTestResults = function() {
  if (this.testCase) {
    return this.testCase.getTestResults();
  }
  return null;
};
