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

    function asyncTestFn(fn) {
      return function(done) {
        // deferred object for signaling completion of asychronous function within globalFn
        var asyncFnDone = webdriver.promise.defer(),
          originalFail = jasmine.getEnv().fail;

        if (fn.length === 0) {
          // function with globalFn not asychronous
          asyncFnDone.fulfill();
        } else if (fn.length > 1) {
          throw Error('Invalid # arguments (' + fn.length + ') within function "' + fnName +'"');
        } else {
          // Add fail method to async done callback and override env fail to
          // reject async done promise
          jasmine.getEnv().fail = asyncFnDone.fulfill.fail = function(userError) {
            asyncFnDone.reject(new Error(userError));
          };
        }

        var flowFinished = flow.execute(function() {
          fn.call(jasmine.getEnv(), asyncFnDone.fulfill);
        });

        webdriver.promise.all([asyncFnDone, flowFinished]).then(function() {
          jasmine.getEnv().fail = originalFail;
          seal(done)();
        }, function(e) {
          jasmine.getEnv().fail = originalFail;
          e.stack = e.stack + '\n==== async task ====\n' + driverError.stack;
          done.fail(e);
        });
      };
    }

    var description, func, timeout;
    switch (fnName) {
      case 'it':
      case 'fit':
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
      case 'beforeAll':
      case 'afterAll':
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
global.fit = wrapInControlFlow(global.fit, 'fit');
global.beforeEach = wrapInControlFlow(global.beforeEach, 'beforeEach');
global.afterEach = wrapInControlFlow(global.afterEach, 'afterEach');
global.beforeAll = wrapInControlFlow(global.beforeAll, 'beforeAll');
global.afterAll = wrapInControlFlow(global.afterAll, 'afterAll');

var originalExpect = global.expect;
global.expect = function(actual) {
  if (actual instanceof webdriver.WebElement) {
    throw 'expect called with WebElement argument, expected a Promise. ' +
        'Did you mean to use .getText()?';
  }
  return originalExpect(actual);
};

/**
 * Creates a matcher wrapper that resolves any promises given for actual and
 * expected values, as well as the `pass` property of the result.
 */
jasmine.Expectation.prototype.wrapCompare = function(name, matcherFactory) {
  return function() {
    var expected = Array.prototype.slice.call(arguments, 0),
      expectation = this,
      matchError = new Error("Failed expectation");

    matchError.stack = matchError.stack.replace(/ +at.+jasminewd.+\n/, '');

    flow.execute(function() {
      return webdriver.promise.when(expectation.actual).then(function(actual) {
        return webdriver.promise.all(expected).then(function(expected) {
          return compare(actual, expected);
        });
      });
    });

    function compare(actual, expected) {
      var args = expected.slice(0);
      args.unshift(actual);

      var matcher = matcherFactory(expectation.util, expectation.customEqualityTesters);
      var matcherCompare = matcher.compare;

      if (expectation.isNot) {
        matcherCompare = matcher.negativeCompare || defaultNegativeCompare;
      }

      var result = matcherCompare.apply(null, args);

      return webdriver.promise.when(result.pass).then(function(pass) {
        var message = "";

        if (!pass) {
          if (!result.message) {
            args.unshift(expectation.isNot);
            args.unshift(name);
            message = expectation.util.buildFailureMessage.apply(null, args);
          } else {
            message = result.message;
          }
        }

        if (expected.length == 1) {
          expected = expected[0];
        }
        var res = {
          matcherName: name,
          passed: pass,
          message: message,
          actual: actual,
          expected: expected,
          error: matchError
        };
        expectation.addExpectationResult(pass, res);
      });

      function defaultNegativeCompare() {
        var result = matcher.compare.apply(null, args);
        if (webdriver.promise.isPromise(result.pass)) {
          result.pass = result.pass.then(function(pass) {
            return !pass;
          });
        } else {
          result.pass = !result.pass;
        }
        return result;
      }
    }
  };
};

// Re-add core matchers so they are wrapped.
jasmine.Expectation.addCoreMatchers(jasmine.matchers);

/**
 * A Jasmine reporter which does nothing but execute the input function
 * on a timeout failure.
 */
var OnTimeoutReporter = function(fn) {
  this.callback = fn;
};

OnTimeoutReporter.prototype.specDone = function(result) {
  if (result.status === 'failed') {
    for (var i = 0; i < result.failedExpectations.length; i++) {
      var failureMessage = result.failedExpectations[i].message;

      if (failureMessage.match(/Timeout/)) {
        this.callback();
      }
    }
  }
};

// On timeout, the flow should be reset. This will prevent webdriver tasks
// from overflowing into the next test and causing it to fail or timeout
// as well. This is done in the reporter instead of an afterEach block
// to ensure that it runs after any afterEach() blocks with webdriver tasks
// get to complete first.
jasmine.getEnv().addReporter(new OnTimeoutReporter(function() {
  console.warn('A Jasmine spec timed out. Resetting the WebDriver Control Flow.');
  console.warn('The last active task was: ');
  console.warn(
      (flow.activeFrame_ && flow.activeFrame_.getPendingTask() ?
          flow.activeFrame_.getPendingTask().toString() : 
          'unknown'));
  flow.reset();
}));
