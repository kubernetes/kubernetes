var common = require('../common');
var assert = common.assert;
var fake = common.fake.create();
var retry = require(common.dir.lib + '/retry');

(function testErrors() {
  var operation = retry.operation();

  var error = new Error('some error');
  var error2 = new Error('some other error');
  operation._errors.push(error);
  operation._errors.push(error2);

  assert.deepEqual(operation.errors(), [error, error2]);
})();

(function testMainErrorReturnsMostFrequentError() {
  var operation = retry.operation();
  var error = new Error('some error');
  var error2 = new Error('some other error');

  operation._errors.push(error);
  operation._errors.push(error2);
  operation._errors.push(error);

  assert.strictEqual(operation.mainError(), error);
})();

(function testMainErrorReturnsLastErrorOnEqualCount() {
  var operation = retry.operation();
  var error = new Error('some error');
  var error2 = new Error('some other error');

  operation._errors.push(error);
  operation._errors.push(error2);

  assert.strictEqual(operation.mainError(), error2);
})();

(function testAttempt() {
  var operation = retry.operation();
  var fn = new Function();

  var timeoutOpts = {
    timeout: 1,
    cb: function() {}
  };
  operation.attempt(fn, timeoutOpts);

  assert.strictEqual(fn, operation._fn);
  assert.strictEqual(timeoutOpts.timeout, operation._operationTimeout);
  assert.strictEqual(timeoutOpts.cb, operation._operationTimeoutCb);
})();

(function testRetry() {
  var times = 3;
  var error = new Error('some error');
  var operation = retry.operation([1, 2, 3]);
  var attempts = 0;

  var finalCallback = fake.callback('finalCallback');
  fake.expectAnytime(finalCallback);

  var fn = function() {
    operation.attempt(function(currentAttempt) {
      attempts++;
      assert.equal(currentAttempt, attempts);
      if (operation.retry(error)) {
        return;
      }

      assert.strictEqual(attempts, 4);
      assert.strictEqual(operation.attempts(), attempts);
      assert.strictEqual(operation.mainError(), error);
      finalCallback();
    });
  };

  fn();
})();