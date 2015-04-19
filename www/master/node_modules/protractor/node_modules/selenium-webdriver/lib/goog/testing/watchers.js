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

/**
 * @fileoverview Simple notifiers for the Closure testing framework.
 *
 * @author johnlenz@google.com (John Lenz)
 */

goog.provide('goog.testing.watchers');


/** @private {!Array.<function()>} */
goog.testing.watchers.resetWatchers_ = [];


/**
 * Fires clock reset watching functions.
 */
goog.testing.watchers.signalClockReset = function() {
  var watchers = goog.testing.watchers.resetWatchers_;
  for (var i = 0; i < watchers.length; i++) {
    goog.testing.watchers.resetWatchers_[i]();
  }
};


/**
 * Enqueues a function to be called when the clock used for setTimeout is reset.
 * @param {function()} fn
 */
goog.testing.watchers.watchClockReset = function(fn) {
  goog.testing.watchers.resetWatchers_.push(fn);
};

