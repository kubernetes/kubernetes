// Copyright 2014 Software Freedom Conservancy. All Rights Reserved.
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
 * @fileoverview Defines common conditions for use with
 * {@link webdriver.WebDriver#wait WebDriver wait}.
 *
 * <p>Sample usage:
 * <code><pre>
 *   driver.get('http://www.google.com/ncr');
 *
 *   var query = driver.wait(until.elementLocated(By.name('q')));
 *   query.sendKeys('webdriver\n');
 *
 *   driver.wait(until.titleIs('webdriver - Google Search'));
 * </pre></code>
 *
 * <p>To define a custom condition, simply call WebDriver.wait with a function
 * that will eventually return a truthy-value (neither null, undefined, false,
 * 0, or the empty string):
 * <code><pre>
 *   driver.wait(function() {
 *     return driver.getTitle().then(function(title) {
 *       return title === 'webdriver - Google Search';
 *     });
 *   }, 1000);
 * </pre></code>
 */

goog.provide('webdriver.until');

goog.require('bot.ErrorCode');
goog.require('goog.array');
goog.require('goog.string');



goog.scope(function() {

var until = webdriver.until;


/**
 * Defines a condition to 
 * @param {string} message A descriptive error message. Should complete the
 *     sentence "Waiting [...]"
 * @param {function(!webdriver.WebDriver): OUT} fn The condition function to
 *     evaluate on each iteration of the wait loop.
 * @constructor
 * @struct
 * @final
 * @template OUT
 */
until.Condition = function(message, fn) {
  /** @private {string} */
  this.description_ = 'Waiting ' + message;

  /** @type {function(!webdriver.WebDriver): OUT} */
  this.fn = fn;
};


/** @return {string} A description of this condition. */
until.Condition.prototype.description = function() {
  return this.description_;
};


/**
 * Creates a condition that will wait until the input driver is able to switch
 * to the designated frame. The target frame may be specified as:
 * <ol>
 *   <li>A numeric index into {@code window.frames} for the currently selected
 *       frame.
 *   <li>A {@link webdriver.WebElement}, which must reference a FRAME or IFRAME
 *       element on the current page.
 *   <li>A locator which may be used to first locate a FRAME or IFRAME on the
 *       current page before attempting to switch to it.
 * </ol>
 *
 * <p>Upon successful resolution of this condition, the driver will be left
 * focused on the new frame.
 *
 * @param {!(number|webdriver.WebElement|
 *           webdriver.Locator|webdriver.By.Hash|
 *           function(!webdriver.WebDriver): !webdriver.WebElement)} frame
 *     The frame identifier.
 * @return {!until.Condition.<boolean>} A new condition.
 */
until.ableToSwitchToFrame = function(frame) {
  var condition;
  if (goog.isNumber(frame) || frame instanceof webdriver.WebElement) {
    condition = attemptToSwitchFrames;
  } else {
    condition = function(driver) {
      var locator =
          /** @type {!(webdriver.Locator|webdriver.By.Hash|Function)} */(frame);
      return driver.findElements(locator).then(function(els) {
        if (els.length) {
          return attemptToSwitchFrames(driver, els[0]);
        }
      });
    };
  }

  return new until.Condition('to be able to switch to frame', condition);

  function attemptToSwitchFrames(driver, frame) {
    return driver.switchTo().frame(frame).then(
        function() { return true; },
        function(e) {
          if (e && e.code !== bot.ErrorCode.NO_SUCH_FRAME) {
            throw e;
          }
        });
  }
};


/**
 * Creates a condition that waits for an alert to be opened. Upon success, the
 * returned promise will be fulfilled with the handle for the opened alert.
 *
 * @return {!until.Condition.<!webdriver.Alert>} The new condition.
 */
until.alertIsPresent = function() {
  return new until.Condition('for alert to be present', function(driver) {
    return driver.switchTo().alert().thenCatch(function(e) {
      if (e && e.code !== bot.ErrorCode.NO_SUCH_ALERT) {
        throw e;
      }
    });
  });
};


/**
 * Creates a condition that will wait for the current page's title to match the
 * given value.
 *
 * @param {string} title The expected page title.
 * @return {!until.Condition.<boolean>} The new condition.
 */
until.titleIs = function(title) {
  return new until.Condition(
      'for title to be ' + goog.string.quote(title),
      function(driver) {
        return driver.getTitle().then(function(t) {
          return t === title;
        });
      });
};


/**
 * Creates a condition that will wait for the current page's title to contain
 * the given substring.
 *
 * @param {string} substr The substring that should be present in the page
 *     title.
 * @return {!until.Condition.<boolean>} The new condition.
 */
until.titleContains = function(substr) {
  return new until.Condition(
      'for title to contain ' + goog.string.quote(substr),
      function(driver) {
        return driver.getTitle().then(function(title) {
          return title.indexOf(substr) !== -1;
        });
      });
};


/**
 * Creates a condition that will wait for the current page's title to match the
 * given regular expression.
 *
 * @param {!RegExp} regex The regular expression to test against.
 * @return {!until.Condition.<boolean>} The new condition.
 */
until.titleMatches = function(regex) {
  return new until.Condition('for title to match ' + regex, function(driver) {
    return driver.getTitle().then(function(title) {
      return regex.test(title);
    });
  });
};


/**
 * Creates a condition that will loop until an element is
 * {@link webdriver.WebDriver#findElement found} with the given locator.
 *
 * @param {!(webdriver.Locator|webdriver.By.Hash|Function)} locator The locator
 *     to use.
 * @return {!until.Condition.<!webdriver.WebElement>} The new condition.
 */
until.elementLocated = function(locator) {
  var locatorStr = goog.isFunction(locator) ? 'function()' : locator + '';
  return new until.Condition('element to be located by ' + locatorStr,
      function(driver) {
        return driver.findElements(locator).then(function(elements) {
          return elements[0];
        });
      });
};


/**
 * Creates a condition that will loop until at least one element is
 * {@link webdriver.WebDriver#findElement found} with the given locator.
 *
 * @param {!(webdriver.Locator|webdriver.By.Hash|Function)} locator The locator
 *     to use.
 * @return {!until.Condition.<!Array.<!webdriver.WebElement>>} The new
 *     condition.
 */
until.elementsLocated = function(locator) {
  var locatorStr = goog.isFunction(locator) ? 'function()' : locator + '';
  return new until.Condition(
      'at least one element to be located by ' + locatorStr,
      function(driver) {
        return driver.findElements(locator).then(function(elements) {
          return elements.length > 0 ? elements : null;
        });
      });
};


/**
 * Creates a condition that will wait for the given element to become stale. An
 * element is considered stale once it is removed from the DOM, or a new page
 * has loaded.
 *
 * @param {!webdriver.WebElement} element The element that should become stale.
 * @return {!until.Condition.<boolean>} The new condition.
 */
until.stalenessOf = function(element) {
  return new until.Condition('element to become stale', function() {
    return element.getTagName().then(
        function() { return false; },
        function(e) {
          if (e.code === bot.ErrorCode.STALE_ELEMENT_REFERENCE) {
            return true;
          }
          throw e;
        });
  });
};


/**
 * Creates a condition that will wait for the given element to become visible.
 *
 * @param {!webdriver.WebElement} element The element to test.
 * @return {!until.Condition.<boolean>} The new condition.
 * @see webdriver.WebDriver#isDisplayed
 */
until.elementIsVisible = function(element) {
  return new until.Condition('until element is visible', function() {
    return element.isDisplayed();
  });
};


/**
 * Creates a condition that will wait for the given element to be in the DOM,
 * yet not visible to the user.
 *
 * @param {!webdriver.WebElement} element The element to test.
 * @return {!until.Condition.<boolean>} The new condition.
 * @see webdriver.WebDriver#isDisplayed
 */
until.elementIsNotVisible = function(element) {
  return new until.Condition('until element is not visible', function() {
    return element.isDisplayed().then(function(v) {
      return !v;
    });
  });
};


/**
 * Creates a condition that will wait for the given element to be enabled.
 *
 * @param {!webdriver.WebElement} element The element to test.
 * @return {!until.Condition.<boolean>} The new condition.
 * @see webdriver.WebDriver#isEnabled
 */
until.elementIsEnabled = function(element) {
  return new until.Condition('until element is enabled', function() {
    return element.isEnabled();
  });
};


/**
 * Creates a condition that will wait for the given element to be disabled.
 *
 * @param {!webdriver.WebElement} element The element to test.
 * @return {!until.Condition.<boolean>} The new condition.
 * @see webdriver.WebDriver#isEnabled
 */
until.elementIsDisabled = function(element) {
  return new until.Condition('until element is disabled', function() {
    return element.isEnabled().then(function(v) {
      return !v;
    });
  });
};


/**
 * Creates a condition that will wait for the given element to be selected.
 * @param {!webdriver.WebElement} element The element to test.
 * @return {!until.Condition.<boolean>} The new condition.
 * @see webdriver.WebDriver#isSelected
 */
until.elementIsSelected = function(element) {
  return new until.Condition('until element is selected', function() {
    return element.isSelected();
  });
};


/**
 * Creates a condition that will wait for the given element to be deselected.
 *
 * @param {!webdriver.WebElement} element The element to test.
 * @return {!until.Condition.<boolean>} The new condition.
 * @see webdriver.WebDriver#isSelected
 */
until.elementIsNotSelected = function(element) {
  return new until.Condition('until element is not selected', function() {
    return element.isSelected().then(function(v) {
      return !v;
    });
  });
};


/**
 * Creates a condition that will wait for the given element's
 * {@link webdriver.WebDriver#getText visible text} to match the given
 * {@code text} exactly.
 *
 * @param {!webdriver.WebElement} element The element to test.
 * @param {string} text The expected text.
 * @return {!until.Condition.<boolean>} The new condition.
 * @see webdriver.WebDriver#getText
 */
until.elementTextIs = function(element, text) {
  return new until.Condition('until element text is', function() {
    return element.getText().then(function(t) {
      return t === text;
    });
  });
};


/**
 * Creates a condition that will wait for the given element's
 * {@link webdriver.WebDriver#getText visible text} to contain the given
 * substring.
 *
 * @param {!webdriver.WebElement} element The element to test.
 * @param {string} substr The substring to search for.
 * @return {!until.Condition.<boolean>} The new condition.
 * @see webdriver.WebDriver#getText
 */
until.elementTextContains = function(element, substr) {
  return new until.Condition('until element text contains', function() {
    return element.getText().then(function(t) {
      return t.indexOf(substr) != -1;
    });
  });
};


/**
 * Creates a condition that will wait for the given element's
 * {@link webdriver.WebDriver#getText visible text} to match a regular
 * expression.
 *
 * @param {!webdriver.WebElement} element The element to test.
 * @param {!RegExp} regex The regular expression to test against.
 * @return {!until.Condition.<boolean>} The new condition.
 * @see webdriver.WebDriver#getText
 */
until.elementTextMatches = function(element, regex) {
  return new until.Condition('until element text matches', function() {
    return element.getText().then(function(t) {
      return regex.test(t);
    });
  });
};
});  // goog.scope
