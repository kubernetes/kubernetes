// Copyright 2013 Software Freedom Conservancy. All Rights Reserved.
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

goog.provide('webdriver.testing.Clock');
goog.provide('webdriver.testing.promise.FlowTester');

goog.require('goog.array');
goog.require('webdriver.promise.ControlFlow');



/**
 * Describes an object that can be used to advance the clock and trigger
 * timeouts registered on the global timing functions.
 * @interface
 */
webdriver.testing.Clock = function() {};


/**
 * Advances the clock.
 * @param {number=} opt_ms The number of milliseconds to advance the clock by.
 *     Defaults to 1 ms.
 */
webdriver.testing.Clock.prototype.tick = function(opt_ms) {};



/**
 * Utility for writing unit tests against a
 * {@link webdriver.promise.ControlFlow}. This class assumes the global
 * timeout functions (e.g. setTimeout) have been replaced with test doubles.
 * These doubles should allow registered timeouts to be triggered by
 * calling {@code clock.tick()}.
 * @param {!webdriver.testing.Clock} clock The fake clock to use.
 * @param {{clearInterval: function(number),
 *          clearTimeout: function(number),
 *          setInterval: function(!Function, number): number,
 *          setTimeout: function(!Function, number): number}} timer
 *     The timer object to use for the application under test.
 * @constructor
 * @implements {goog.disposable.IDisposable}
 */
webdriver.testing.promise.FlowTester = function(clock, timer) {

  var self = this;

  /** @private {!webdriver.testing.Clock} */
  this.clock_ = clock;

  /**
   * @private {!Array.<{isIdle: boolean,
   *                    errors: !Array.<!Error>,
   *                    flow: !webdriver.promise.ControlFlow}>}
   */
  this.allFlows_ = [];

  /** @private {!webdriver.promise.ControlFlow} */
  this.flow_ = createFlow();

  /** @private {!webdriver.promise.ControlFlow} */
  this.originalFlow_ = webdriver.promise.controlFlow();
  webdriver.promise.setDefaultFlow(this.flow_);

  /**
   * @private {function(function(!webdriver.promise.ControlFlow)):
   *               !webdriver.promise.Promise}
   */
  this.originalCreateFlow_ = webdriver.promise.createFlow;

  /**
   * @param {function(!webdriver.promise.ControlFlow)} callback The entry point
   *     to the newly created flow.
   * @return {!webdriver.promise.Promise} .
   */
  webdriver.promise.createFlow = function(callback) {
    var flow = createFlow();
    return flow.execute(function() {
      return callback(flow);
    });
  };

  function createFlow() {
    var record = {
      isIdle: true,
      errors: [],
      flow: new webdriver.promise.ControlFlow(timer).
          on(webdriver.promise.ControlFlow.EventType.IDLE, function() {
            record.isIdle = true;
          }).
          on(webdriver.promise.ControlFlow.EventType.SCHEDULE_TASK, function() {
            record.isIdle = false;
          }).
          on(webdriver.promise.ControlFlow.EventType.UNCAUGHT_EXCEPTION,
             function(e) {
               record.isIdle = true;
               record.errors.push(e);
             })
    };
    self.allFlows_.push(record);
    return record.flow;
  }
};


/** @private {boolean} */
webdriver.testing.promise.FlowTester.prototype.isDisposed_ = false;


/** @override */
webdriver.testing.promise.FlowTester.prototype.isDisposed = function() {
  return this.isDisposed_;
};


/**
 * Disposes of this instance, restoring the default control flow object.
 * @override
 */
webdriver.testing.promise.FlowTester.prototype.dispose = function() {
  if (!this.isDisposed_) {
    goog.array.forEach(this.allFlows_, function(record) {
      record.flow.reset();
    });
    webdriver.promise.setDefaultFlow(this.originalFlow_);
    webdriver.promise.createFlow = this.originalCreateFlow_;
    this.isDisposed_ = true;
  }
};


/**
 * @param {!Array} messages The array to join into a single error message.
 * @throws {Error} An error with a message formatted from {@code messages}.
 * @private
 */
webdriver.testing.promise.FlowTester.prototype.throwWithMessages_ = function(
    messages) {
  throw Error(messages.join('\n--------\n') + '\n === done ===');
};


/**
 * Verifies that all created flows are idle and completed without any errors.
 * If a specific flow object is provided, will only check that flow.
 * @param {webdriver.promise.ControlFlow=} opt_flow The specific flow to check.
 *     If not specified, will verify against all flows.
 */
webdriver.testing.promise.FlowTester.prototype.verifySuccess = function(
    opt_flow) {
  var messages = [];
  var foundFlow = false;

  goog.array.forEach(this.allFlows_, function(record, index) {
    if (!opt_flow || opt_flow === record.flow) {
      foundFlow = true;
      if (record.errors.length) {
        messages = goog.array.concat(
            messages, 'Uncaught errors for flow #' + index,
            goog.array.map(record.errors, function(error) {
              if (!error) {
                error = error + '';
              }
              return error.stack || error.message || String(error);
            }));
      }

      if (!record.isIdle) {
        messages.push('Flow #' + index + ' is not idle');
      }
    }
  });

  if (opt_flow && !foundFlow) {
    messages.push('Specified flow not found!');
  }

  if (messages.length) {
    this.throwWithMessages_(messages);
  }
};


/**
 * Verifies that all flows are idle and at least one reported a single failure.
 * If a specific flow object is provided, will only check that flow.
 * @param {webdriver.promise.ControlFlow=} opt_flow The flow expected to have
 *     generated the error.
 */
webdriver.testing.promise.FlowTester.prototype.verifyFailure = function(
    opt_flow) {
  var messages = [];
  var foundFlow = false;
  var foundAFailure = false;

  goog.array.forEach(this.allFlows_, function(record, index) {
    if (!opt_flow || opt_flow === record.flow) {
      foundFlow = true;

      if (!record.isIdle) {
        messages.push('Flow #' + index + ' is not idle');
      }

      if (record.errors.length) {
        foundAFailure = true;

        if (record.errors.length > 1) {
          messages = goog.array.concat(
              messages, 'Flow #' + index + ' had multiple errors: ',
              record.errors);
        }
      }

    }
  });

  if (opt_flow && !foundFlow) {
    messages.push('Specified flow not found!');
  }

  if (!foundAFailure) {
    messages.push('No failures found!');
  }

  if (messages.length) {
    this.throwWithMessages_(messages);
  }
};


/**
 * @param {webdriver.promise.ControlFlow=} opt_flow The specific control flow
 *     to check. If not specified, will implicitly verify there was a single
 *     error across all flows.
 * @return {Error} The error reported by the application.
 * @throws {Error} If the application is not finished, or did not report
 *     exactly one error.
 */
webdriver.testing.promise.FlowTester.prototype.getFailure = function(
    opt_flow) {
  this.verifyFailure(opt_flow);

  var errors = [];
  goog.array.forEach(this.allFlows_, function(record) {
    if (!opt_flow || opt_flow === record.flow) {
      errors = goog.array.concat(errors, record.errors);
    }
  });

  if (errors.length > 1) {
    this.throwWithMessages_(goog.array.concat(
        'There was more than one failure', errors));
  }

  return errors[0];
};


/**
 * Advances the clock so the {@link webdriver.promise.ControlFlow}'s event
 * loop will run once.
 */
webdriver.testing.promise.FlowTester.prototype.turnEventLoop = function() {
  this.clock_.tick(webdriver.promise.ControlFlow.EVENT_LOOP_FREQUENCY);
};


/**
 * @param {webdriver.promise.ControlFlow=} opt_flow The specific flow to check.
 *     If not specified, will verify all flows are still running.
 * @throws {Error} If the application is not running.
 */
webdriver.testing.promise.FlowTester.prototype.assertStillRunning = function(
    opt_flow) {
  var messages = [];
  var foundFlow = false;

  goog.array.forEach(this.allFlows_, function(record, index) {
    if (!opt_flow || opt_flow === record.flow) {
      foundFlow = true;
      if (record.isIdle) {
        messages.push('Flow #' + index + ' is idle');
      }

      if (record.errors.length) {
        messages = goog.array.concat(
            messages, 'Uncaught errors for flow #' + index, record.errors);
      }
    }
  });

  if (opt_flow && !foundFlow) {
    messages.push('Specified flow not found!');
  }

  if (messages.length) {
    this.throwWithMessages_(messages);
  }
};


/**
 * Runs the application, turning its event loop until it is expected to have
 * shutdown (as indicated by having no more frames, or no frames with pending
 * tasks).
 */
webdriver.testing.promise.FlowTester.prototype.run = function() {
  var flow = this.flow_;
  var self = this;
  var done = false;
  var shouldBeDone = false;

  while (!done) {
    this.turnEventLoop();
    if (shouldBeDone) {
      assertIsDone();
    } else {
      determineIfShouldBeDone();
    }

    // If the event loop generated an unhandled promise, it won't be reported
    // until one more turn of the JS event loop, so we need to tick the
    // clock once more. This is necessary for our tests to simulate a real
    // JS environment.
    this.clock_.tick();
  }

  function assertIsDone() {
    // Shutdown is done in one extra turn of the event loop.
    self.clock_.tick();
    done = goog.array.every(self.allFlows_, function(record) {
      return record.isIdle;
    });

    if (!done) {
      goog.array.forEach(self.allFlows_, function(record) {
        if (record.flow.activeFrame_) {
          // Not done yet, but there are no frames left.  This can happen if the
          // very first scheduled task was scheduled inside of a promise
          // callback. Turn the event loop one more time; the flow should detect
          // that it is now finished and start the shutdown procedure.  Don't
          // recurse here since we could go into an infinite loop if the flow is
          // broken.
          self.turnEventLoop();
          self.clock_.tick();
        }
      });
    }

    if (!done) {
      throw Error('Should be done now: ' + flow.getSchedule());
    }
  }

  function determineIfShouldBeDone() {
    shouldBeDone = flow === webdriver.promise.controlFlow() &&
        goog.array.every(self.allFlows_, function(record) {
          return !record.flow.activeFrame_;
        });
  }
};
