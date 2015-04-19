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
 * @fileoverview A map of listeners that provides utility functions to
 * deal with listeners on an event target. Used by
 * {@code goog.events.EventTarget}.
 *
 * WARNING: Do not use this class from outside goog.events package.
 *
 * @visibility {//closure/goog/bin/sizetests:__pkg__}
 * @visibility {//closure/goog/events:__pkg__}
 * @visibility {//closure/goog/labs/events:__pkg__}
 */

goog.provide('goog.events.ListenerMap');

goog.require('goog.array');
goog.require('goog.events.Listener');
goog.require('goog.object');



/**
 * Creates a new listener map.
 * @param {EventTarget|goog.events.Listenable} src The src object.
 * @constructor
 * @final
 */
goog.events.ListenerMap = function(src) {
  /** @type {EventTarget|goog.events.Listenable} */
  this.src = src;

  /**
   * Maps of event type to an array of listeners.
   * @type {Object.<string, !Array.<!goog.events.Listener>>}
   */
  this.listeners = {};

  /**
   * The count of types in this map that have registered listeners.
   * @private {number}
   */
  this.typeCount_ = 0;
};


/**
 * @return {number} The count of event types in this map that actually
 *     have registered listeners.
 */
goog.events.ListenerMap.prototype.getTypeCount = function() {
  return this.typeCount_;
};


/**
 * @return {number} Total number of registered listeners.
 */
goog.events.ListenerMap.prototype.getListenerCount = function() {
  var count = 0;
  for (var type in this.listeners) {
    count += this.listeners[type].length;
  }
  return count;
};


/**
 * Adds an event listener. A listener can only be added once to an
 * object and if it is added again the key for the listener is
 * returned.
 *
 * Note that a one-off listener will not change an existing listener,
 * if any. On the other hand a normal listener will change existing
 * one-off listener to become a normal listener.
 *
 * @param {string|!goog.events.EventId} type The listener event type.
 * @param {!Function} listener This listener callback method.
 * @param {boolean} callOnce Whether the listener is a one-off
 *     listener.
 * @param {boolean=} opt_useCapture The capture mode of the listener.
 * @param {Object=} opt_listenerScope Object in whose scope to call the
 *     listener.
 * @return {goog.events.ListenableKey} Unique key for the listener.
 */
goog.events.ListenerMap.prototype.add = function(
    type, listener, callOnce, opt_useCapture, opt_listenerScope) {
  var typeStr = type.toString();
  var listenerArray = this.listeners[typeStr];
  if (!listenerArray) {
    listenerArray = this.listeners[typeStr] = [];
    this.typeCount_++;
  }

  var listenerObj;
  var index = goog.events.ListenerMap.findListenerIndex_(
      listenerArray, listener, opt_useCapture, opt_listenerScope);
  if (index > -1) {
    listenerObj = listenerArray[index];
    if (!callOnce) {
      // Ensure that, if there is an existing callOnce listener, it is no
      // longer a callOnce listener.
      listenerObj.callOnce = false;
    }
  } else {
    listenerObj = new goog.events.Listener(
        listener, null, this.src, typeStr, !!opt_useCapture, opt_listenerScope);
    listenerObj.callOnce = callOnce;
    listenerArray.push(listenerObj);
  }
  return listenerObj;
};


/**
 * Removes a matching listener.
 * @param {string|!goog.events.EventId} type The listener event type.
 * @param {!Function} listener This listener callback method.
 * @param {boolean=} opt_useCapture The capture mode of the listener.
 * @param {Object=} opt_listenerScope Object in whose scope to call the
 *     listener.
 * @return {boolean} Whether any listener was removed.
 */
goog.events.ListenerMap.prototype.remove = function(
    type, listener, opt_useCapture, opt_listenerScope) {
  var typeStr = type.toString();
  if (!(typeStr in this.listeners)) {
    return false;
  }

  var listenerArray = this.listeners[typeStr];
  var index = goog.events.ListenerMap.findListenerIndex_(
      listenerArray, listener, opt_useCapture, opt_listenerScope);
  if (index > -1) {
    var listenerObj = listenerArray[index];
    listenerObj.markAsRemoved();
    goog.array.removeAt(listenerArray, index);
    if (listenerArray.length == 0) {
      delete this.listeners[typeStr];
      this.typeCount_--;
    }
    return true;
  }
  return false;
};


/**
 * Removes the given listener object.
 * @param {goog.events.ListenableKey} listener The listener to remove.
 * @return {boolean} Whether the listener is removed.
 */
goog.events.ListenerMap.prototype.removeByKey = function(listener) {
  var type = listener.type;
  if (!(type in this.listeners)) {
    return false;
  }

  var removed = goog.array.remove(this.listeners[type], listener);
  if (removed) {
    listener.markAsRemoved();
    if (this.listeners[type].length == 0) {
      delete this.listeners[type];
      this.typeCount_--;
    }
  }
  return removed;
};


/**
 * Removes all listeners from this map. If opt_type is provided, only
 * listeners that match the given type are removed.
 * @param {string|!goog.events.EventId=} opt_type Type of event to remove.
 * @return {number} Number of listeners removed.
 */
goog.events.ListenerMap.prototype.removeAll = function(opt_type) {
  var typeStr = opt_type && opt_type.toString();
  var count = 0;
  for (var type in this.listeners) {
    if (!typeStr || type == typeStr) {
      var listenerArray = this.listeners[type];
      for (var i = 0; i < listenerArray.length; i++) {
        ++count;
        listenerArray[i].markAsRemoved();
      }
      delete this.listeners[type];
      this.typeCount_--;
    }
  }
  return count;
};


/**
 * Gets all listeners that match the given type and capture mode. The
 * returned array is a copy (but the listener objects are not).
 * @param {string|!goog.events.EventId} type The type of the listeners
 *     to retrieve.
 * @param {boolean} capture The capture mode of the listeners to retrieve.
 * @return {!Array.<goog.events.ListenableKey>} An array of matching
 *     listeners.
 */
goog.events.ListenerMap.prototype.getListeners = function(type, capture) {
  var listenerArray = this.listeners[type.toString()];
  var rv = [];
  if (listenerArray) {
    for (var i = 0; i < listenerArray.length; ++i) {
      var listenerObj = listenerArray[i];
      if (listenerObj.capture == capture) {
        rv.push(listenerObj);
      }
    }
  }
  return rv;
};


/**
 * Gets the goog.events.ListenableKey for the event or null if no such
 * listener is in use.
 *
 * @param {string|!goog.events.EventId} type The type of the listener
 *     to retrieve.
 * @param {!Function} listener The listener function to get.
 * @param {boolean} capture Whether the listener is a capturing listener.
 * @param {Object=} opt_listenerScope Object in whose scope to call the
 *     listener.
 * @return {goog.events.ListenableKey} the found listener or null if not found.
 */
goog.events.ListenerMap.prototype.getListener = function(
    type, listener, capture, opt_listenerScope) {
  var listenerArray = this.listeners[type.toString()];
  var i = -1;
  if (listenerArray) {
    i = goog.events.ListenerMap.findListenerIndex_(
        listenerArray, listener, capture, opt_listenerScope);
  }
  return i > -1 ? listenerArray[i] : null;
};


/**
 * Whether there is a matching listener. If either the type or capture
 * parameters are unspecified, the function will match on the
 * remaining criteria.
 *
 * @param {string|!goog.events.EventId=} opt_type The type of the listener.
 * @param {boolean=} opt_capture The capture mode of the listener.
 * @return {boolean} Whether there is an active listener matching
 *     the requested type and/or capture phase.
 */
goog.events.ListenerMap.prototype.hasListener = function(
    opt_type, opt_capture) {
  var hasType = goog.isDef(opt_type);
  var typeStr = hasType ? opt_type.toString() : '';
  var hasCapture = goog.isDef(opt_capture);

  return goog.object.some(
      this.listeners, function(listenerArray, type) {
        for (var i = 0; i < listenerArray.length; ++i) {
          if ((!hasType || listenerArray[i].type == typeStr) &&
              (!hasCapture || listenerArray[i].capture == opt_capture)) {
            return true;
          }
        }

        return false;
      });
};


/**
 * Finds the index of a matching goog.events.Listener in the given
 * listenerArray.
 * @param {!Array.<!goog.events.Listener>} listenerArray Array of listener.
 * @param {!Function} listener The listener function.
 * @param {boolean=} opt_useCapture The capture flag for the listener.
 * @param {Object=} opt_listenerScope The listener scope.
 * @return {number} The index of the matching listener within the
 *     listenerArray.
 * @private
 */
goog.events.ListenerMap.findListenerIndex_ = function(
    listenerArray, listener, opt_useCapture, opt_listenerScope) {
  for (var i = 0; i < listenerArray.length; ++i) {
    var listenerObj = listenerArray[i];
    if (!listenerObj.removed &&
        listenerObj.listener == listener &&
        listenerObj.capture == !!opt_useCapture &&
        listenerObj.handler == opt_listenerScope) {
      return i;
    }
  }
  return -1;
};
