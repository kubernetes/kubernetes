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
 * @fileoverview Factory methods for the supported locator strategies.
 */

goog.provide('webdriver.By');
goog.provide('webdriver.Locator');
goog.provide('webdriver.Locator.Strategy');

goog.require('goog.array');
goog.require('goog.object');
goog.require('goog.string');



/**
 * An element locator.
 * @param {string} using The type of strategy to use for this locator.
 * @param {string} value The search target of this locator.
 * @constructor
 */
webdriver.Locator = function(using, value) {

  /**
   * The search strategy to use when searching for an element.
   * @type {string}
   */
  this.using = using;

  /**
   * The search target for this locator.
   * @type {string}
   */
  this.value = value;
};


/**
 * Creates a factory function for a {@link webdriver.Locator}.
 * @param {string} type The type of locator for the factory.
 * @return {function(string): !webdriver.Locator} The new factory function.
 * @private
 */
webdriver.Locator.factory_ = function(type) {
  return function(value) {
    return new webdriver.Locator(type, value);
  };
};


/**
 * A collection of factory functions for creating {@link webdriver.Locator}
 * instances.
 */
webdriver.By = {};
// Exported to the global scope for legacy reasons.
goog.exportSymbol('By', webdriver.By);


/**
 * Short-hand expressions for the primary element locator strategies.
 * For example the following two statements are equivalent:
 * <code><pre>
 * var e1 = driver.findElement(webdriver.By.id('foo'));
 * var e2 = driver.findElement({id: 'foo'});
 * </pre></code>
 *
 * <p>Care should be taken when using JavaScript minifiers (such as the
 * Closure compiler), as locator hashes will always be parsed using
 * the un-obfuscated properties listed below.
 *
 * @typedef {(
 *     {className: string}|
 *     {css: string}|
 *     {id: string}|
 *     {js: string}|
 *     {linkText: string}|
 *     {name: string}|
 *     {partialLinkText: string}|
 *     {tagName: string}|
 *     {xpath: string})}
 */
webdriver.By.Hash;


/**
 * Locates elements that have a specific class name. The returned locator
 * is equivalent to searching for elements with the CSS selector ".clazz".
 *
 * @param {string} className The class name to search for.
 * @return {!webdriver.Locator} The new locator.
 * @see http://www.w3.org/TR/2011/WD-html5-20110525/elements.html#classes
 * @see http://www.w3.org/TR/CSS2/selector.html#class-html
 */
webdriver.By.className = webdriver.Locator.factory_('class name');


/**
 * Locates elements using a CSS selector. For browsers that do not support
 * CSS selectors, WebDriver implementations may return an
 * {@link bot.Error.State.INVALID_SELECTOR invalid selector} error. An
 * implementation may, however, emulate the CSS selector API.
 *
 * @param {string} selector The CSS selector to use.
 * @return {!webdriver.Locator} The new locator.
 * @see http://www.w3.org/TR/CSS2/selector.html
 */
webdriver.By.css = webdriver.Locator.factory_('css selector');


/**
 * Locates an element by its ID.
 *
 * @param {string} id The ID to search for.
 * @return {!webdriver.Locator} The new locator.
 */
webdriver.By.id = webdriver.Locator.factory_('id');


/**
 * Locates link elements whose {@link webdriver.WebElement#getText visible
 * text} matches the given string.
 *
 * @param {string} text The link text to search for.
 * @return {!webdriver.Locator} The new locator.
 */
webdriver.By.linkText = webdriver.Locator.factory_('link text');


/**
 * Locates an elements by evaluating a
 * {@link webdriver.WebDriver#executeScript JavaScript expression}.
 * The result of this expression must be an element or list of elements.
 *
 * @param {!(string|Function)} script The script to execute.
 * @param {...*} var_args The arguments to pass to the script.
 * @return {function(!webdriver.WebDriver): !webdriver.promise.Promise} A new,
 *     JavaScript-based locator function.
 */
webdriver.By.js = function(script, var_args) {
  var args = goog.array.slice(arguments, 0);
  return function(driver) {
    return driver.executeScript.apply(driver, args);
  };
};


/**
 * Locates elements whose {@code name} attribute has the given value.
 *
 * @param {string} name The name attribute to search for.
 * @return {!webdriver.Locator} The new locator.
 */
webdriver.By.name = webdriver.Locator.factory_('name');


/**
 * Locates link elements whose {@link webdriver.WebElement#getText visible
 * text} contains the given substring.
 *
 * @param {string} text The substring to check for in a link's visible text.
 * @return {!webdriver.Locator} The new locator.
 */
webdriver.By.partialLinkText = webdriver.Locator.factory_(
    'partial link text');


/**
 * Locates elements with a given tag name. The returned locator is
 * equivalent to using the {@code getElementsByTagName} DOM function.
 *
 * @param {string} text The substring to check for in a link's visible text.
 * @return {!webdriver.Locator} The new locator.
 * @see http://www.w3.org/TR/REC-DOM-Level-1/level-one-core.html
 */
webdriver.By.tagName = webdriver.Locator.factory_('tag name');


/**
 * Locates elements matching a XPath selector. Care should be taken when
 * using an XPath selector with a {@link webdriver.WebElement} as WebDriver
 * will respect the context in the specified in the selector. For example,
 * given the selector {@code "//div"}, WebDriver will search from the
 * document root regardless of whether the locator was used with a
 * WebElement.
 *
 * @param {string} xpath The XPath selector to use.
 * @return {!webdriver.Locator} The new locator.
 * @see http://www.w3.org/TR/xpath/
 */
webdriver.By.xpath = webdriver.Locator.factory_('xpath');


/**
 * Maps {@link webdriver.By.Hash} keys to the appropriate factory function.
 * @type {!Object.<string, function(string): !(Function|webdriver.Locator)>}
 * @const
 */
webdriver.Locator.Strategy = {
  'className': webdriver.By.className,
  'css': webdriver.By.css,
  'id': webdriver.By.id,
  'js': webdriver.By.js,
  'linkText': webdriver.By.linkText,
  'name': webdriver.By.name,
  'partialLinkText': webdriver.By.partialLinkText,
  'tagName': webdriver.By.tagName,
  'xpath': webdriver.By.xpath
};


/**
 * Verifies that a {@code value} is a valid locator to use for searching for
 * elements on the page.
 *
 * @param {*} value The value to check is a valid locator.
 * @return {!(webdriver.Locator|Function)} A valid locator object or function.
 * @throws {TypeError} If the given value is an invalid locator.
 */
webdriver.Locator.checkLocator = function(value) {
  if (goog.isFunction(value) || value instanceof webdriver.Locator) {
    return value;
  }
  for (var key in value) {
    if (value.hasOwnProperty(key) &&
        webdriver.Locator.Strategy.hasOwnProperty(key)) {
      return webdriver.Locator.Strategy[key](value[key]);
    }
  }
  throw new TypeError('Invalid locator');
};



/** @override */
webdriver.Locator.prototype.toString = function() {
  return 'By.' + this.using.replace(/ ([a-z])/g, function(all, match) {
    return match.toUpperCase();
  }) + '(' + goog.string.quote(this.value) + ')';
};
