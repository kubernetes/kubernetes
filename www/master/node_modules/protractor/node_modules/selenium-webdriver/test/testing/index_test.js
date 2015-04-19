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

'use strict';

var assert = require('assert');
var promise = require('../..').promise;

var test = require('../../testing');

describe('Mocha Integration', function() {

  describe('beforeEach properly binds "this"', function() {
    beforeEach(function() { this.x = 1; });
    test.beforeEach(function() { this.x = 2; });
    it('', function() { assert.equal(this.x, 2); });
  });

  describe('afterEach properly binds "this"', function() {
    it('', function() { this.x = 1; });
    test.afterEach(function() { this.x = 2; });
    afterEach(function() { assert.equal(this.x, 2); });
  });

  describe('it properly binds "this"', function() {
    beforeEach(function() { this.x = 1; });
    test.it('', function() { this.x = 2; });
    afterEach(function() { assert.equal(this.x, 2); });
  });

  describe('it properly allows timeouts and cancels control flow', function() {
    var timeoutErr, flowReset;

    beforeEach(function() {
      flowReset = false;
      promise.controlFlow().on(promise.ControlFlow.EventType.RESET, function() {
        flowReset = true;
      });
    });

    test.it('', function() {
      var mochaCallback = this.runnable().callback.mochaCallback;
      this.runnable().callback.mochaCallback = function(err) {
        timeoutErr = err;
        // We do not pass along the arguments because we want this it block
        // to always pass, we apply the tests that ensure the timeout was
        // successfully called and that the controlFlow promise were reset
        // in the afterEach block.
        return mochaCallback.apply(this);
      };

      this.timeout(1000);
      var unresolvedPromise = promise.defer();
      return unresolvedPromise.promise;
    });

    afterEach(function() {
      assert.equal(
        flowReset,
        true,
        'the controlFlow for the test block should be cancelled on timeout'
      );

      assert.equal(
        timeoutErr instanceof Error,
        true,
        'the testing error should be propegated back up to the mocha test runner'
      );

      assert.equal(
        timeoutErr.message,
        'timeout of 1000ms exceeded'
      );
    });
  });
});
