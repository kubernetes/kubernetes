// Copyright 2013 Selenium committers
// Copyright 2013 Software Freedom Conservancy
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//     You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Provides wrappers around the following global functions from
 * <a href="http://visionmedia.github.io/mocha/">Mocha's BDD interface</a>:
 * <ul>
 *   <li>after
 *   <li>afterEach
 *   <li>before
 *   <li>beforeEach
 *   <li>it
 *   <li>it.only
 *   <li>it.skip
 *   <li>xit
 * </ul>
 *
 * <p>The provided wrappers leverage the {@link webdriver.promise.ControlFlow}
 * to simplify writing asynchronous tests:
 * <pre><code>
 * var By = require('selenium-webdriver').By,
 *     until = require('selenium-webdriver').until,
 *     firefox = require('selenium-webdriver/firefox'),
 *     test = require('selenium-webdriver/testing');
 *
 * test.describe('Google Search', function() {
 *   var driver;
 *
 *   test.before(function() {
 *     driver = new firefox.Driver();
 *   });
 *
 *   test.after(function() {
 *     driver.quit();
 *   });
 *
 *   test.it('should append query to title', function() {
 *     driver.get('http://www.google.com/ncr');
 *     driver.findElement(By.name('q')).sendKeys('webdriver');
 *     driver.findElement(By.name('btnG')).click();
 *     driver.wait(until.titleIs('webdriver - Google Search'), 1000);
 *   });
 * });
 * </code></pre>
 *
 * <p>You may conditionally suppress a test function using the exported
 * "ignore" function. If the provided predicate returns true, the attached
 * test case will be skipped:
 * <pre><code>
 *   test.ignore(maybe()).it('is flaky', function() {
 *     if (Math.random() < 0.5) throw Error();
 *   });
 *
 *   function maybe() { return Math.random() < 0.5; }
 * </code></pre>
 */

var promise = require('..').promise;
var flow = promise.controlFlow();


/**
 * Wraps a function so that all passed arguments are ignored.
 * @param {!Function} fn The function to wrap.
 * @return {!Function} The wrapped function.
 */
function seal(fn) {
  return function() {
    fn();
  };
}


/**
 * Wraps a function on Mocha's BDD interface so it runs inside a
 * webdriver.promise.ControlFlow and waits for the flow to complete before
 * continuing.
 * @param {!Function} globalFn The function to wrap.
 * @return {!Function} The new function.
 */
function wrapped(globalFn) {
  return function() {
    if (arguments.length === 1) {
      return globalFn(asyncTestFn(arguments[0]));
    }
    else if (arguments.length === 2) {
      return globalFn(arguments[0], asyncTestFn(arguments[1]));
    }
    else {
      throw Error('Invalid # arguments: ' + arguments.length);
    }
  };

  function asyncTestFn(fn) {
    var ret = function(done) {
      function cleanupBeforeCallback() {
        flow.reset();
        return cleanupBeforeCallback.mochaCallback.apply(this, arguments);
      }
      // We set this as an attribute of the callback function to allow us to
      // test this properly.
      cleanupBeforeCallback.mochaCallback = this.runnable().callback;

      this.runnable().callback = cleanupBeforeCallback;

      var testFn = fn.bind(this);
      flow.execute(function() {
        var done = promise.defer();
        promise.asap(testFn(done.reject), done.fulfill, done.reject);
        return done.promise;
      }).then(seal(done), done);
    };

    ret.toString = function() {
      return fn.toString();
    };

    return ret;
  }
}


/**
 * Ignores the test chained to this function if the provided predicate returns
 * true.
 * @param {function(): boolean} predicateFn A predicate to call to determine
 *     if the test should be suppressed. This function MUST be synchronous.
 * @return {!Object} An object with wrapped versions of {@link #it()} and
 *     {@link #describe()} that ignore tests as indicated by the predicate.
 */
function ignore(predicateFn) {
  var describe = wrap(exports.xdescribe, exports.describe);
  describe.only = wrap(exports.xdescribe, exports.describe.only);

  var it = wrap(exports.xit, exports.it);
  it.only = wrap(exports.xit, exports.it.only);

  return {
    describe: describe,
    it: it
  };

  function wrap(onSkip, onRun) {
    return function(title, fn) {
      if (predicateFn()) {
        onSkip(title, fn);
      } else {
        onRun(title, fn);
      }
    };
  }
}


// PUBLIC API

/**
 * Registers a new test suite.
 * @param {string} name The suite name.
 * @param {function()=} fn The suite function, or {@code undefined} to define
 *     a pending test suite.
 */
exports.describe = global.describe;

/**
 * Defines a suppressed test suite.
 * @param {string} name The suite name.
 * @param {function()=} fn The suite function, or {@code undefined} to define
 *     a pending test suite.
 */
exports.xdescribe = global.xdescribe;
exports.describe.skip = global.describe.skip;

/**
 * Register a function to call after the current suite finishes.
 * @param {function()} fn .
 */
exports.after = wrapped(global.after);

/**
 * Register a function to call after each test in a suite.
 * @param {function()} fn .
 */
exports.afterEach = wrapped(global.afterEach);

/**
 * Register a function to call before the current suite starts.
 * @param {function()} fn .
 */
exports.before = wrapped(global.before);

/**
 * Register a function to call before each test in a suite.
 * @param {function()} fn .
 */
exports.beforeEach = wrapped(global.beforeEach);

/**
 * Add a test to the current suite.
 * @param {string} name The test name.
 * @param {function()=} fn The test function, or {@code undefined} to define
 *     a pending test case.
 */
exports.it = wrapped(global.it);

/**
 * An alias for {@link #it()} that flags the test as the only one that should
 * be run within the current suite.
 * @param {string} name The test name.
 * @param {function()=} fn The test function, or {@code undefined} to define
 *     a pending test case.
 */
exports.iit = exports.it.only = wrapped(global.it.only);

/**
 * Adds a test to the current suite while suppressing it so it is not run.
 * @param {string} name The test name.
 * @param {function()=} fn The test function, or {@code undefined} to define
 *     a pending test case.
 */
exports.xit = exports.it.skip = wrapped(global.xit);

exports.ignore = ignore;
