// Copyright 2010 The Closure Library Authors. All Rights Reserved.
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
 * @fileoverview An interface that all mocks should share.
 * @author nicksantos@google.com (Nick Santos)
 */

goog.provide('goog.testing.MockInterface');



/** @interface */
goog.testing.MockInterface = function() {};


/**
 * Write down all the expected functions that have been called on the
 * mock so far. From here on out, future function calls will be
 * compared against this list.
 */
goog.testing.MockInterface.prototype.$replay = function() {};


/**
 * Reset the mock.
 */
goog.testing.MockInterface.prototype.$reset = function() {};


/**
 * Assert that the expected function calls match the actual calls.
 */
goog.testing.MockInterface.prototype.$verify = function() {};
