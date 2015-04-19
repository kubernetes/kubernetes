// Copyright 2005 The Closure Library Authors. All Rights Reserved.
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
 * @fileoverview Listener object.
 * @see ../demos/events.html
 */

goog.provide('goog.events.Listener');

goog.require('goog.events.ListenableKey');



/**
 * Simple class that stores information about a listener
 * @param {!Function} listener Callback function.
 * @param {Function} proxy Wrapper for the listener that patches the event.
 * @param {EventTarget|goog.events.Listenable} src Source object for
 *     the event.
 * @param {string} type Event type.
 * @param {boolean} capture Whether in capture or bubble phase.
 * @param {Object=} opt_handler Object in whose context to execute the callback.
 * @implements {goog.events.ListenableKey}
 * @constructor
 */
goog.events.Listener = function(
    listener, proxy, src, type, capture, opt_handler) {
  if (goog.events.Listener.ENABLE_MONITORING) {
    this.creationStack = new Error().stack;
  }

  /**
   * Callback function.
   * @type {Function}
   */
  this.listener = listener;

  /**
   * A wrapper over the original listener. This is used solely to
   * handle native browser events (it is used to simulate the capture
   * phase and to patch the event object).
   * @type {Function}
   */
  this.proxy = proxy;

  /**
   * Object or node that callback is listening to
   * @type {EventTarget|goog.events.Listenable}
   */
  this.src = src;

  /**
   * The event type.
   * @const {string}
   */
  this.type = type;

  /**
   * Whether the listener is being called in the capture or bubble phase
   * @const {boolean}
   */
  this.capture = !!capture;

  /**
   * Optional object whose context to execute the listener in
   * @type {Object|undefined}
   */
  this.handler = opt_handler;

  /**
   * The key of the listener.
   * @const {number}
   * @override
   */
  this.key = goog.events.ListenableKey.reserveKey();

  /**
   * Whether to remove the listener after it has been called.
   * @type {boolean}
   */
  this.callOnce = false;

  /**
   * Whether the listener has been removed.
   * @type {boolean}
   */
  this.removed = false;
};


/**
 * @define {boolean} Whether to enable the monitoring of the
 *     goog.events.Listener instances. Switching on the monitoring is only
 *     recommended for debugging because it has a significant impact on
 *     performance and memory usage. If switched off, the monitoring code
 *     compiles down to 0 bytes.
 */
goog.define('goog.events.Listener.ENABLE_MONITORING', false);


/**
 * If monitoring the goog.events.Listener instances is enabled, stores the
 * creation stack trace of the Disposable instance.
 * @type {string}
 */
goog.events.Listener.prototype.creationStack;


/**
 * Marks this listener as removed. This also remove references held by
 * this listener object (such as listener and event source).
 */
goog.events.Listener.prototype.markAsRemoved = function() {
  this.removed = true;
  this.listener = null;
  this.proxy = null;
  this.src = null;
  this.handler = null;
};
