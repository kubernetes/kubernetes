// Copyright 2012 Software Freedom Conservancy. All Rights Reserved.
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
 * @fileoverview A utility class for working with test windows.
 */

goog.provide('webdriver.testing.Window');

goog.require('goog.string');
goog.require('webdriver.promise.Promise');



/**
 * Class for managing a window.
 *
 * <p>This class is implemented as a promise so consumers may register
 * callbacks on it to handle situations where the window fails to open.
 *
 * For example:
 * <pre><code>
 *   var testWindow = webdriver.testing.Window.create(driver);
 *   // Throw a custom error when the window fails to open.
 *   testWindow.thenCatch(function(e) {
 *     throw Error('Failed to open test window: ' + e);
 *   });
 * </code></pre>
 *
 * @param {!webdriver.WebDriver} driver The driver to use.
 * @param {(string|!webdriver.promise.Promise)} handle Either the managed
 *     window's handle, or a promise that will resolve to it.
 * @param {(Window|webdriver.promise.Promise)=} opt_window The raw window
 *     object, if available.
 * @constructor
 * @extends {webdriver.promise.Promise}
 */
webdriver.testing.Window = function(driver, handle, opt_window) {
  webdriver.promise.Promise.call(this);

  /** @private {!webdriver.WebDriver} */
  this.driver_ = driver;

  /** @private {!webdriver.promise.Promise} */
  this.handle_ = webdriver.promise.when(handle);

  /** @private {!webdriver.promise.Promise} */
  this.window_ = webdriver.promise.when(opt_window);
};
goog.inherits(webdriver.testing.Window, webdriver.promise.Promise);


/**
 * Default amount of time, in milliseconds, to wait for a new window to open.
 * @type {number}
 * @const
 */
webdriver.testing.Window.DEFAULT_OPEN_TIMEOUT = 2000;


/**
 * Window running this script. Lazily initialized the first time it is requested
 * from {@link webdriver.testing.Window.findWindow}.
 * @private {webdriver.testing.Window}
 */
webdriver.testing.Window.currentWindow_ = null;


/**
 * Opens and focuses on a new window.
 *
 * @param {!webdriver.WebDriver} driver The driver to use.
 * @param {?{width: number, height: number}=} opt_size The desired size for
 *     the new window.
 * @param {number=} opt_timeout How long, in milliseconds, to wait for the new
 *     window to open. Defaults to
 *     {@link webdriver.testing.Window.DEFAULT_OPEN_TIMEOUT}.
 * @return {!webdriver.testing.Window} The new window.
 */
webdriver.testing.Window.create = function(driver, opt_size, opt_timeout) {
  var windowPromise = webdriver.promise.defer();
  var handle = driver.call(function() {
    var features = [
      'location=yes',
      'titlebar=yes'
    ];

    if (opt_size) {
      features.push(
          'width=' + opt_size.width,
          'height=' + opt_size.height);
    }

    var name = goog.string.getRandomString();
    windowPromise.fulfill(window.open('', name, features.join(',')));

    driver.wait(function() {
      return driver.switchTo().window(name).then(
          function() { return true; },
          function() { return false; });
    }, opt_timeout || webdriver.testing.Window.DEFAULT_OPEN_TIMEOUT);
    return driver.getWindowHandle();
  });

  return new webdriver.testing.Window(driver, handle, windowPromise);
};


/**
 * Changes focus to the topmost window for the provided DOM window.
 *
 * @param {!webdriver.WebDriver} driver The driver to use.
 * @param {Window=} opt_window The window to search for. Defaults to the window
 *     running this script.
 * @return {!webdriver.testing.Window} The located window.
 */
webdriver.testing.Window.focusOnWindow = function(driver, opt_window) {
  var name = goog.string.getRandomString();
  var win = opt_window ? opt_window.top : window;

  var ret;
  if (win === window && webdriver.testing.Window.currentWindow_) {
    ret = webdriver.testing.Window.currentWindow_;
    webdriver.testing.Window.currentWindow_.focus();
  } else {
    win.name = name;
    ret = new webdriver.testing.Window(driver,
        driver.switchTo().window(name).
            then(goog.bind(driver.getWindowHandle, driver)),
        win);
    if (win === window) {
      webdriver.testing.Window.currentWindow_ = ret;
    }
  }
  return ret;
};


/** @override */
webdriver.testing.Window.prototype.cancel = function() {
  return this.handle_.cancel();
};


/** @override */
webdriver.testing.Window.prototype.then = function(callback, errback) {
  return this.handle_.then(callback, errback);
};


/**
 * Focuses the wrapped driver on the window managed by this class.
 * @return {!webdriver.promise.Promise} A promise that will be resolved
 *     when this command has completed.
 */
webdriver.testing.Window.prototype.focus = function() {
  return this.driver_.switchTo().window(this.handle_);
};


/**
 * Focuses on and closes the managed window. The driver <em>must</em> be
 * focused on another window before issuing any further commands.
 * @return {!webdriver.promise.Promise} A promise that will be resolved
 *     when this command has completed.
 */
webdriver.testing.Window.prototype.close = function() {
  var self = this;
  return this.window_.then(function(win) {
    if (win) {
      win.close();
    } else {
      return self.focus().then(goog.bind(self.driver_.close, self.driver_));
    }
  });
};


/**
 * Retrieves the current size of this window.
 * @return {!webdriver.promise.Promise} A promise that resolves to the size of
 *     this window as a {width:number, height:number} object.
 */
webdriver.testing.Window.prototype.getSize = function() {
  var driver = this.driver_;
  return this.focus().then(function() {
    return driver.manage().window().getSize();
  });
};


/**
 * Sets the size of this window.
 * @param {number} width The desired width, in pixels.
 * @param {number} height The desired height, in pixels.
 * @return {!webdriver.promise.Promise} A promise that resolves when the
 *     command has completed.
 */
webdriver.testing.Window.prototype.setSize = function(width, height) {
  var driver = this.driver_;
  return this.focus().then(function() {
    return driver.manage().window().setSize(width, height);
  });
};


/**
 * Retrieves the current position of this window, in pixels relative to the
 * upper left corner of the screen.
 * @return {!webdriver.promise.Promise} A promise that resolves to the
 *     position of this window as a {x:number, y:number} object.
 */
webdriver.testing.Window.prototype.getPosition = function() {
  var driver = this.driver_;
  return this.focus().then(function() {
    return driver.manage().window().getPosition();
  });
};


/**
 * Repositions this window.
 * @param {number} x The desired horizontal position, in pixels from the left
 *     side of the screen.
 * @param {number} y The desired vertical position, in pixels from the top
 *     of the screen.
 * @return {!webdriver.promise.Promise} A promise that resolves when the
 *     command has completed.
 */
webdriver.testing.Window.prototype.setPosition = function(x, y) {
  var driver = this.driver_;
  return this.focus().then(function() {
    return driver.manage().window().setPosition(x, y);
  });
};
