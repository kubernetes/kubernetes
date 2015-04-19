/**
 * Adapts Jasmine-Node tests to work better with WebDriverJS. Borrows
 * heavily from the mocha WebDriverJS adapter at
 * https://code.google.com/p/selenium/source/browse/javascript/node/selenium-webdriver/testing/index.js
 */

var webdriver = require('selenium-webdriver');

var flow = webdriver.promise.controlFlow();

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
 * Validates that the parameter is a function.
 * @param {Object} functionToValidate The function to validate.
 * @throws {Error}
 * @return {Object} The original parameter.
 */
function validateFunction(functionToValidate) {
  if (functionToValidate && Object.prototype.toString.call(functionToValidate) === '[object Function]') {
    return functionToValidate;
  } else {
    throw Error(functionToValidate + ' is not a function');
  }
}

/**
 * Validates that the parameter is a number.
 * @param {Object} numberToValidate The number to validate.
 * @throws {Error}
 * @return {Object} The original number.
 */
function validateNumber(numberToValidate) {
  if (!isNaN(numberToValidate)) {
    return numberToValidate;
  } else {
    throw Error(numberToValidate + ' is not a number');
  }
}

/**
 * Validates that the parameter is a string.
 * @param {Object} stringToValidate The string to validate.
 * @throws {Error}
 * @return {Object} The original string.
 */
function validateString(stringtoValidate) {
  if (typeof stringtoValidate == 'string' || stringtoValidate instanceof String) {
    return stringtoValidate;
  } else {
    throw Error(stringtoValidate + ' is not a string');
  }
}

/**
 * Wraps a function so it runs inside a webdriver.promise.ControlFlow and
 * waits for the flow to complete before continuing.
 * @param {!Function} globalFn The function to wrap.
 * @return {!Function} The new function.
 */
function wrapInControlFlow(globalFn, fnName) {
  return function() {
    var driverError = new Error();
    driverError.stack = driverError.stack.replace(/ +at.+jasminewd.+\n/, '');

    function asyncTestFn(fn, desc) {
      return function(done) {
        var desc_ = 'Asynchronous test function: ' + fnName + '(';
        if (desc) {
          desc_ += '"' + desc + '"';
        }
        desc_ += ')';

        // deferred object for signaling completion of asychronous function within globalFn
        var asyncFnDone = webdriver.promise.defer();

        if (fn.length === 0) {
          // function with globalFn not asychronous
          asyncFnDone.fulfill();
        } else if (fn.length > 1) {
          throw Error('Invalid # arguments (' + fn.length + ') within function "' + fnName +'"');
        }

        var flowFinished = flow.execute(function() {
          fn.call(jasmine.getEnv().currentSpec, function(userError) {
            if (userError) {
              asyncFnDone.reject(new Error(userError));
            } else {
              asyncFnDone.fulfill();
            }
          });
        }, desc_);

        webdriver.promise.all([asyncFnDone, flowFinished]).then(function() {
          seal(done)();
        }, function(e) {
          e.stack = e.stack + '==== async task ====\n' + driverError.stack;
          done(e);
        });
      };
    }

    var description, func, timeout;
    switch (fnName) {
      case 'it':
      case 'iit':
        description = validateString(arguments[0]);
        func = validateFunction(arguments[1]);
        if (!arguments[2]) {
          globalFn(description, asyncTestFn(func));
        } else {
          timeout = validateNumber(arguments[2]);
          globalFn(description, asyncTestFn(func), timeout);
        }
        break;
      case 'beforeEach':
      case 'afterEach':
        func = validateFunction(arguments[0]);
        if (!arguments[1]) {
          globalFn(asyncTestFn(func));
        } else {
          timeout = validateNumber(arguments[1]);
          globalFn(asyncTestFn(func), timeout);
        }
        break;
      default:
        throw Error('invalid function: ' + fnName);
    }
  };
}

global.it = wrapInControlFlow(global.it, 'it');
global.iit = wrapInControlFlow(global.iit, 'iit');
global.beforeEach = wrapInControlFlow(global.beforeEach, 'beforeEach');
global.afterEach = wrapInControlFlow(global.afterEach, 'afterEach');


/**
 * Wrap a Jasmine matcher function so that it can take webdriverJS promises.
 * @param {!Function} matcher The matcher function to wrap.
 * @param {webdriver.promise.Promise} actualPromise The promise which will
 *     resolve to the actual value being tested.
 * @param {boolean} not Whether this is being called with 'not' active.
 */
function wrapMatcher(matcher, actualPromise, not) {
  return function() {
    var originalArgs = arguments;
    var matchError = new Error("Failed expectation");
    matchError.stack = matchError.stack.replace(/ +at.+jasminewd.+\n/, '');
    actualPromise.then(function(actual) {
      var expected = originalArgs[0];

      var expectation = originalExpect(actual);
      if (not) {
        expectation = expectation.not;
      }
      var originalAddMatcherResult = expectation.spec.addMatcherResult;
      var error = matchError;
      expectation.spec.addMatcherResult = function(result) {
        result.trace = error;
        jasmine.Spec.prototype.addMatcherResult.call(this, result);
      };

      if (webdriver.promise.isPromise(expected)) {
        if (originalArgs.length > 1) {
          throw error('Multi-argument matchers with promises are not ' +
              'supported.');
        }
        expected.then(function(exp) {
          expectation[matcher].apply(expectation, [exp]);
          expectation.spec.addMatcherResult = originalAddMatcherResult;
        });
      } else {
        expectation.spec.addMatcherResult = function(result) {
          result.trace = error;
          originalAddMatcherResult.call(this, result);
        };
        expectation[matcher].apply(expectation, originalArgs);
        expectation.spec.addMatcherResult = originalAddMatcherResult;
      }
    });
  };
}

/**
 * Return a chained set of matcher functions which will be evaluated
 * after actualPromise is resolved.
 * @param {webdriver.promise.Promise} actualPromise The promise which will
 *     resolve to the actual value being tested.
 */
function promiseMatchers(actualPromise) {
  var promises = {not: {}};
  var env = jasmine.getEnv();
  var matchersClass = env.currentSpec.matchersClass || env.matchersClass;

  for (var matcher in matchersClass.prototype) {
    promises[matcher] = wrapMatcher(matcher, actualPromise, false);
    promises.not[matcher] = wrapMatcher(matcher, actualPromise, true);
  }

  return promises;
}

var originalExpect = global.expect;

global.expect = function(actual) {
  if (actual instanceof webdriver.WebElement) {
    throw 'expect called with WebElement argument, expected a Promise. ' +
        'Did you mean to use .getText()?';
  }
  if (webdriver.promise.isPromise(actual)) {
    return promiseMatchers(actual);
  } else {
    return originalExpect(actual);
  }
};

// Wrap internal Jasmine function to allow custom matchers
// to return promises that resolve to truthy or falsy values
var originalMatcherFn = jasmine.Matchers.matcherFn_;
jasmine.Matchers.matcherFn_ = function(matcherName, matcherFunction) {
  var matcherFnThis = this;
  var matcherFnArgs = jasmine.util.argsToArray(arguments);
  return function() {
    var matcherThis = this;
    var matcherArgs = jasmine.util.argsToArray(arguments);
    var result = matcherFunction.apply(this, arguments);

    if (webdriver.promise.isPromise(result)) {
      result.then(function(resolution) {
        matcherFnArgs[1] = function() {
          return resolution;
        };
        originalMatcherFn.apply(matcherFnThis, matcherFnArgs).
            apply(matcherThis, matcherArgs);
      });
    } else {
      originalMatcherFn.apply(matcherFnThis, matcherFnArgs).
          apply(matcherThis, matcherArgs);
    }
  };
};

/**
 * A Jasmine reporter which does nothing but execute the input function
 * on a timeout failure.
 */
var OnTimeoutReporter = function(fn) {
  this.callback = fn;
};

OnTimeoutReporter.prototype.reportRunnerStarting = function() {};
OnTimeoutReporter.prototype.reportRunnerResults = function() {};
OnTimeoutReporter.prototype.reportSuiteResults = function() {};
OnTimeoutReporter.prototype.reportSpecStarting = function() {};
OnTimeoutReporter.prototype.reportSpecResults = function(spec) {
  if (!spec.results().passed()) {
    var result = spec.results();
    var failureItem = null;

    var items_length = result.getItems().length;
    for (var i = 0; i < items_length; i++) {
      if (result.getItems()[i].passed_ === false) {
        failureItem = result.getItems()[i];

        var jasmineTimeoutRegexp =
            /timed out after \d+ msec waiting for spec to complete/;
        if (failureItem.toString().match(jasmineTimeoutRegexp)) {
          this.callback();
        }
      }
    }
  }
};
OnTimeoutReporter.prototype.log = function() {};

// On timeout, the flow should be reset. This will prevent webdriver tasks
// from overflowing into the next test and causing it to fail or timeout
// as well. This is done in the reporter instead of an afterEach block
// to ensure that it runs after any afterEach() blocks with webdriver tasks
// get to complete first.
jasmine.getEnv().addReporter(new OnTimeoutReporter(function() {
  console.warn('A Jasmine spec timed out. Resetting the WebDriver Control Flow.');
  console.warn('The last active task was: ');
  console.warn(
      (flow.activeFrame_  && flow.activeFrame_.getPendingTask() ?
          flow.activeFrame_.getPendingTask().toString() : 
          'unknown'));
  flow.reset();
}));
