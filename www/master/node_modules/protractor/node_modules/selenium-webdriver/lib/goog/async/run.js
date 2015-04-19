// Copyright 2013 The Closure Library Authors. All Rights Reserved.
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

goog.provide('goog.async.run');

goog.require('goog.async.nextTick');
goog.require('goog.async.throwException');
goog.require('goog.testing.watchers');


/**
 * Fires the provided callback just before the current callstack unwinds, or as
 * soon as possible after the current JS execution context.
 * @param {function(this:THIS)} callback
 * @param {THIS=} opt_context Object to use as the "this value" when calling
 *     the provided function.
 * @template THIS
 */
goog.async.run = function(callback, opt_context) {
  if (!goog.async.run.schedule_) {
    goog.async.run.initializeRunner_();
  }
  if (!goog.async.run.workQueueScheduled_) {
    // Nothing is currently scheduled, schedule it now.
    goog.async.run.schedule_();
    goog.async.run.workQueueScheduled_ = true;
  }

  goog.async.run.workQueue_.push(
      new goog.async.run.WorkItem_(callback, opt_context));
};


/**
 * Initializes the function to use to process the work queue.
 * @private
 */
goog.async.run.initializeRunner_ = function() {
  // If native Promises are available in the browser, just schedule the callback
  // on a fulfilled promise, which is specified to be async, but as fast as
  // possible.
  if (goog.global.Promise && goog.global.Promise.resolve) {
    var promise = goog.global.Promise.resolve();
    goog.async.run.schedule_ = function() {
      promise.then(goog.async.run.processWorkQueue);
    };
  } else {
    goog.async.run.schedule_ = function() {
      goog.async.nextTick(goog.async.run.processWorkQueue);
    };
  }
};


/**
 * Forces goog.async.run to use nextTick instead of Promise.
 *
 * This should only be done in unit tests. It's useful because MockClock
 * replaces nextTick, but not the browser Promise implementation, so it allows
 * Promise-based code to be tested with MockClock.
 */
goog.async.run.forceNextTick = function() {
  goog.async.run.schedule_ = function() {
    goog.async.nextTick(goog.async.run.processWorkQueue);
  };
};


/**
 * The function used to schedule work asynchronousely.
 * @private {function()}
 */
goog.async.run.schedule_;


/** @private {boolean} */
goog.async.run.workQueueScheduled_ = false;


/** @private {!Array.<!goog.async.run.WorkItem_>} */
goog.async.run.workQueue_ = [];


if (goog.DEBUG) {
  /**
   * Reset the event queue.
   * @private
   */
  goog.async.run.resetQueue_ = function() {
    goog.async.run.workQueueScheduled_ = false;
    goog.async.run.workQueue_ = [];
  };

  // If there is a clock implemenation in use for testing
  // and it is reset, reset the queue.
  goog.testing.watchers.watchClockReset(goog.async.run.resetQueue_);
}


/**
 * Run any pending goog.async.run work items. This function is not intended
 * for general use, but for use by entry point handlers to run items ahead of
 * goog.async.nextTick.
 */
goog.async.run.processWorkQueue = function() {
  // NOTE: additional work queue items may be pushed while processing.
  while (goog.async.run.workQueue_.length) {
    // Don't let the work queue grow indefinitely.
    var workItems = goog.async.run.workQueue_;
    goog.async.run.workQueue_ = [];
    for (var i = 0; i < workItems.length; i++) {
      var workItem = workItems[i];
      try {
        workItem.fn.call(workItem.scope);
      } catch (e) {
        goog.async.throwException(e);
      }
    }
  }

  // There are no more work items, reset the work queue.
  goog.async.run.workQueueScheduled_ = false;
};



/**
 * @constructor
 * @final
 * @struct
 * @private
 *
 * @param {function()} fn
 * @param {Object|null|undefined} scope
 */
goog.async.run.WorkItem_ = function(fn, scope) {
  /** @const */ this.fn = fn;
  /** @const */ this.scope = scope;
};
