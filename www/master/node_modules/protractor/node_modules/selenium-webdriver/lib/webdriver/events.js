// Copyright 2011 Software Freedom Conservancy. All Rights Reserved.
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

/**
 * @fileoverview A light weight event system modeled after Node's EventEmitter.
 */

goog.provide('webdriver.EventEmitter');



/**
 * Object that can emit events for others to listen for. This is used instead
 * of Closure's event system because it is much more light weight. The API is
 * based on Node's EventEmitters.
 * @constructor
 */
webdriver.EventEmitter = function() {
  /**
   * Map of events to registered listeners.
   * @private {!Object.<!Array.<{fn: !Function, oneshot: boolean,
   *                             scope: (Object|undefined)}>>}
   */
  this.events_ = {};
};


/**
 * Fires an event and calls all listeners.
 * @param {string} type The type of event to emit.
 * @param {...*} var_args Any arguments to pass to each listener.
 */
webdriver.EventEmitter.prototype.emit = function(type, var_args) {
  var args = Array.prototype.slice.call(arguments, 1);
  var listeners = this.events_[type];
  if (!listeners) {
    return;
  }
  for (var i = 0; i < listeners.length;) {
    var listener = listeners[i];
    listener.fn.apply(listener.scope, args);
    if (listeners[i] === listener) {
      if (listeners[i].oneshot) {
        listeners.splice(i, 1);
      } else {
        i += 1;
      }
    }
  }
};


/**
 * Returns a mutable list of listeners for a specific type of event.
 * @param {string} type The type of event to retrieve the listeners for.
 * @return {!Array.<{fn: !Function, oneshot: boolean,
 *                   scope: (Object|undefined)}>} The registered listeners for
 *     the given event type.
 */
webdriver.EventEmitter.prototype.listeners = function(type) {
  var listeners = this.events_[type];
  if (!listeners) {
    listeners = this.events_[type] = [];
  }
  return listeners;
};


/**
 * Registers a listener.
 * @param {string} type The type of event to listen for.
 * @param {!Function} listenerFn The function to invoke when the event is fired.
 * @param {Object=} opt_scope The object in whose scope to invoke the listener.
 * @param {boolean=} opt_oneshot Whether the listener should be removed after
 *    the first event is fired.
 * @return {!webdriver.EventEmitter} A self reference.
 * @private
 */
webdriver.EventEmitter.prototype.addListener_ = function(type, listenerFn,
    opt_scope, opt_oneshot) {
  var listeners = this.listeners(type);
  var n = listeners.length;
  for (var i = 0; i < n; ++i) {
    if (listeners[i].fn == listenerFn) {
      return this;
    }
  }

  listeners.push({
    fn: listenerFn,
    scope: opt_scope,
    oneshot: !!opt_oneshot
  });
  return this;
};


/**
 * Registers a listener.
 * @param {string} type The type of event to listen for.
 * @param {!Function} listenerFn The function to invoke when the event is fired.
 * @param {Object=} opt_scope The object in whose scope to invoke the listener.
 * @return {!webdriver.EventEmitter} A self reference.
 */
webdriver.EventEmitter.prototype.addListener = function(type, listenerFn,
    opt_scope) {
  return this.addListener_(type, listenerFn, opt_scope);
};


/**
 * Registers a one-time listener which will be called only the first time an
 * event is emitted, after which it will be removed.
 * @param {string} type The type of event to listen for.
 * @param {!Function} listenerFn The function to invoke when the event is fired.
 * @param {Object=} opt_scope The object in whose scope to invoke the listener.
 * @return {!webdriver.EventEmitter} A self reference.
 */
webdriver.EventEmitter.prototype.once = function(type, listenerFn, opt_scope) {
  return this.addListener_(type, listenerFn, opt_scope, true);
};


/**
 * An alias for {@code #addListener()}.
 * @param {string} type The type of event to listen for.
 * @param {!Function} listenerFn The function to invoke when the event is fired.
 * @param {Object=} opt_scope The object in whose scope to invoke the listener.
 * @return {!webdriver.EventEmitter} A self reference.
 */
webdriver.EventEmitter.prototype.on =
    webdriver.EventEmitter.prototype.addListener;


/**
 * Removes a previously registered event listener.
 * @param {string} type The type of event to unregister.
 * @param {!Function} listenerFn The handler function to remove.
 * @return {!webdriver.EventEmitter} A self reference.
 */
webdriver.EventEmitter.prototype.removeListener = function(type, listenerFn) {
  var listeners = this.events_[type];
  if (listeners) {
    var n = listeners.length;
    for (var i = 0; i < n; ++i) {
      if (listeners[i].fn == listenerFn) {
        listeners.splice(i, 1);
        return this;
      }
    }
  }
  return this;
};


/**
 * Removes all listeners for a specific type of event. If no event is
 * specified, all listeners across all types will be removed.
 * @param {string=} opt_type The type of event to remove listeners from.
 * @return {!webdriver.EventEmitter} A self reference.
 */
webdriver.EventEmitter.prototype.removeAllListeners = function(opt_type) {
  goog.isDef(opt_type) ? delete this.events_[opt_type] : this.events_ = {};
  return this;
};
