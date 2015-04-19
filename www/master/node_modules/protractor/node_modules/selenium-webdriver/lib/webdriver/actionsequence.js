// Copyright 2012 Selenium comitters
// Copyright 2012 Software Freedom Conservancy
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

goog.provide('webdriver.ActionSequence');

goog.require('goog.array');
goog.require('webdriver.Button');
goog.require('webdriver.Command');
goog.require('webdriver.CommandName');
goog.require('webdriver.Key');



/**
 * Class for defining sequences of complex user interactions. Each sequence
 * will not be executed until {@link #perform} is called.
 *
 * <p>Example:<pre><code>
 *   new webdriver.ActionSequence(driver).
 *       keyDown(webdriver.Key.SHIFT).
 *       click(element1).
 *       click(element2).
 *       dragAndDrop(element3, element4).
 *       keyUp(webdriver.Key.SHIFT).
 *       perform();
 * </pre></code>
 *
 * @param {!webdriver.WebDriver} driver The driver instance to use.
 * @constructor
 */
webdriver.ActionSequence = function(driver) {

  /** @private {!webdriver.WebDriver} */
  this.driver_ = driver;

  /** @private {!Array.<{description: string, command: !webdriver.Command}>} */
  this.actions_ = [];
};


/**
 * Schedules an action to be executed each time {@link #perform} is called on
 * this instance.
 * @param {string} description A description of the command.
 * @param {!webdriver.Command} command The command.
 * @private
 */
webdriver.ActionSequence.prototype.schedule_ = function(description, command) {
  this.actions_.push({
    description: description,
    command: command
  });
};


/**
 * Executes this action sequence.
 * @return {!webdriver.promise.Promise} A promise that will be resolved once
 *     this sequence has completed.
 */
webdriver.ActionSequence.prototype.perform = function() {
  // Make a protected copy of the scheduled actions. This will protect against
  // users defining additional commands before this sequence is actually
  // executed.
  var actions = goog.array.clone(this.actions_);
  var driver = this.driver_;
  return driver.controlFlow().execute(function() {
    goog.array.forEach(actions, function(action) {
      driver.schedule(action.command, action.description);
    });
  }, 'ActionSequence.perform');
};


/**
 * Moves the mouse.  The location to move to may be specified in terms of the
 * mouse's current location, an offset relative to the top-left corner of an
 * element, or an element (in which case the middle of the element is used).
 * @param {(!webdriver.WebElement|{x: number, y: number})} location The
 *     location to drag to, as either another WebElement or an offset in pixels.
 * @param {{x: number, y: number}=} opt_offset If the target {@code location}
 *     is defined as a {@link webdriver.WebElement}, this parameter defines an
 *     offset within that element. The offset should be specified in pixels
 *     relative to the top-left corner of the element's bounding box. If
 *     omitted, the element's center will be used as the target offset.
 * @return {!webdriver.ActionSequence} A self reference.
 */
webdriver.ActionSequence.prototype.mouseMove = function(location, opt_offset) {
  var command = new webdriver.Command(webdriver.CommandName.MOVE_TO);

  if (goog.isNumber(location.x)) {
    setOffset(/** @type {{x: number, y: number}} */(location));
  } else {
    // The interactions API expect the element ID to be encoded as a simple
    // string, not the usual JSON object.
    var id = /** @type {!webdriver.WebElement} */ (location).getId().
        then(function(value) {
          return value['ELEMENT'];
        });
    command.setParameter('element', id);
    if (opt_offset) {
      setOffset(opt_offset);
    }
  }

  this.schedule_('mouseMove', command);
  return this;

  /** @param {{x: number, y: number}} offset The offset to use. */
  function setOffset(offset) {
    command.setParameter('xoffset', offset.x || 0);
    command.setParameter('yoffset', offset.y || 0);
  }
};


/**
 * Schedules a mouse action.
 * @param {string} description A simple descriptive label for the scheduled
 *     action.
 * @param {!webdriver.CommandName} commandName The name of the command.
 * @param {(webdriver.WebElement|webdriver.Button)=} opt_elementOrButton Either
 *     the element to interact with or the button to click with.
 *     Defaults to {@link webdriver.Button.LEFT} if neither an element nor
 *     button is specified.
 * @param {webdriver.Button=} opt_button The button to use. Defaults to
 *     {@link webdriver.Button.LEFT}. Ignored if the previous argument is
 *     provided as a button.
 * @return {!webdriver.ActionSequence} A self reference.
 * @private
 */
webdriver.ActionSequence.prototype.scheduleMouseAction_ = function(
    description, commandName, opt_elementOrButton, opt_button) {
  var button;
  if (goog.isNumber(opt_elementOrButton)) {
    button = opt_elementOrButton;
  } else {
    if (opt_elementOrButton) {
      this.mouseMove(
          /** @type {!webdriver.WebElement} */ (opt_elementOrButton));
    }
    button = goog.isDef(opt_button) ? opt_button : webdriver.Button.LEFT;
  }

  var command = new webdriver.Command(commandName).
      setParameter('button', button);
  this.schedule_(description, command);
  return this;
};


/**
 * Presses a mouse button. The mouse button will not be released until
 * {@link #mouseUp} is called, regardless of whether that call is made in this
 * sequence or another. The behavior for out-of-order events (e.g. mouseDown,
 * click) is undefined.
 *
 * <p>If an element is provided, the mouse will first be moved to the center
 * of that element. This is equivalent to:
 * <pre><code>sequence.mouseMove(element).mouseDown()</code></pre>
 *
 * <p>Warning: this method currently only supports the left mouse button. See
 * http://code.google.com/p/selenium/issues/detail?id=4047
 *
 * @param {(webdriver.WebElement|webdriver.Button)=} opt_elementOrButton Either
 *     the element to interact with or the button to click with.
 *     Defaults to {@link webdriver.Button.LEFT} if neither an element nor
 *     button is specified.
 * @param {webdriver.Button=} opt_button The button to use. Defaults to
 *     {@link webdriver.Button.LEFT}. Ignored if a button is provided as the
 *     first argument.
 * @return {!webdriver.ActionSequence} A self reference.
 */
webdriver.ActionSequence.prototype.mouseDown = function(opt_elementOrButton,
                                                        opt_button) {
  return this.scheduleMouseAction_('mouseDown',
      webdriver.CommandName.MOUSE_DOWN, opt_elementOrButton, opt_button);
};


/**
 * Releases a mouse button. Behavior is undefined for calling this function
 * without a previous call to {@link #mouseDown}.
 *
 * <p>If an element is provided, the mouse will first be moved to the center
 * of that element. This is equivalent to:
 * <pre><code>sequence.mouseMove(element).mouseUp()</code></pre>
 *
 * <p>Warning: this method currently only supports the left mouse button. See
 * http://code.google.com/p/selenium/issues/detail?id=4047
 *
 * @param {(webdriver.WebElement|webdriver.Button)=} opt_elementOrButton Either
 *     the element to interact with or the button to click with.
 *     Defaults to {@link webdriver.Button.LEFT} if neither an element nor
 *     button is specified.
 * @param {webdriver.Button=} opt_button The button to use. Defaults to
 *     {@link webdriver.Button.LEFT}. Ignored if a button is provided as the
 *     first argument.
 * @return {!webdriver.ActionSequence} A self reference.
 */
webdriver.ActionSequence.prototype.mouseUp = function(opt_elementOrButton,
                                                      opt_button) {
  return this.scheduleMouseAction_('mouseUp',
      webdriver.CommandName.MOUSE_UP, opt_elementOrButton, opt_button);
};


/**
 * Convenience function for performing a "drag and drop" manuever. The target
 * element may be moved to the location of another element, or by an offset (in
 * pixels).
 * @param {!webdriver.WebElement} element The element to drag.
 * @param {(!webdriver.WebElement|{x: number, y: number})} location The
 *     location to drag to, either as another WebElement or an offset in pixels.
 * @return {!webdriver.ActionSequence} A self reference.
 */
webdriver.ActionSequence.prototype.dragAndDrop = function(element, location) {
  return this.mouseDown(element).mouseMove(location).mouseUp();
};


/**
 * Clicks a mouse button.
 *
 * <p>If an element is provided, the mouse will first be moved to the center
 * of that element. This is equivalent to:
 * <pre><code>sequence.mouseMove(element).click()</code></pre>
 *
 * @param {(webdriver.WebElement|webdriver.Button)=} opt_elementOrButton Either
 *     the element to interact with or the button to click with.
 *     Defaults to {@link webdriver.Button.LEFT} if neither an element nor
 *     button is specified.
 * @param {webdriver.Button=} opt_button The button to use. Defaults to
 *     {@link webdriver.Button.LEFT}. Ignored if a button is provided as the
 *     first argument.
 * @return {!webdriver.ActionSequence} A self reference.
 */
webdriver.ActionSequence.prototype.click = function(opt_elementOrButton,
                                                    opt_button) {
  return this.scheduleMouseAction_('click',
      webdriver.CommandName.CLICK, opt_elementOrButton, opt_button);
};


/**
 * Double-clicks a mouse button.
 *
 * <p>If an element is provided, the mouse will first be moved to the center of
 * that element. This is equivalent to:
 * <pre><code>sequence.mouseMove(element).doubleClick()</code></pre>
 *
 * <p>Warning: this method currently only supports the left mouse button. See
 * http://code.google.com/p/selenium/issues/detail?id=4047
 *
 * @param {(webdriver.WebElement|webdriver.Button)=} opt_elementOrButton Either
 *     the element to interact with or the button to click with.
 *     Defaults to {@link webdriver.Button.LEFT} if neither an element nor
 *     button is specified.
 * @param {webdriver.Button=} opt_button The button to use. Defaults to
 *     {@link webdriver.Button.LEFT}. Ignored if a button is provided as the
 *     first argument.
 * @return {!webdriver.ActionSequence} A self reference.
 */
webdriver.ActionSequence.prototype.doubleClick = function(opt_elementOrButton,
                                                          opt_button) {
  return this.scheduleMouseAction_('doubleClick',
      webdriver.CommandName.DOUBLE_CLICK, opt_elementOrButton, opt_button);
};


/**
 * Schedules a keyboard action.
 * @param {string} description A simple descriptive label for the scheduled
 *     action.
 * @param {!Array.<(string|!webdriver.Key)>} keys The keys to send.
 * @return {!webdriver.ActionSequence} A self reference.
 * @private
 */
webdriver.ActionSequence.prototype.scheduleKeyboardAction_ = function(
    description, keys) {
  var command =
      new webdriver.Command(webdriver.CommandName.SEND_KEYS_TO_ACTIVE_ELEMENT).
          setParameter('value', keys);
  this.schedule_(description, command);
  return this;
};


/**
 * Checks that a key is a modifier key.
 * @param {!webdriver.Key} key The key to check.
 * @throws {Error} If the key is not a modifier key.
 * @private
 */
webdriver.ActionSequence.checkModifierKey_ = function(key) {
  if (key !== webdriver.Key.ALT && key !== webdriver.Key.CONTROL &&
      key !== webdriver.Key.SHIFT && key !== webdriver.Key.COMMAND) {
    throw Error('Not a modifier key');
  }
};


/**
 * Performs a modifier key press. The modifier key is <em>not released</em>
 * until {@link #keyUp} or {@link #sendKeys} is called. The key press will be
 * targetted at the currently focused element.
 * @param {!webdriver.Key} key The modifier key to push. Must be one of
 *     {ALT, CONTROL, SHIFT, COMMAND, META}.
 * @return {!webdriver.ActionSequence} A self reference.
 * @throws {Error} If the key is not a valid modifier key.
 */
webdriver.ActionSequence.prototype.keyDown = function(key) {
  webdriver.ActionSequence.checkModifierKey_(key);
  return this.scheduleKeyboardAction_('keyDown', [key]);
};


/**
 * Performs a modifier key release. The release is targetted at the currently
 * focused element.
 * @param {!webdriver.Key} key The modifier key to release. Must be one of
 *     {ALT, CONTROL, SHIFT, COMMAND, META}.
 * @return {!webdriver.ActionSequence} A self reference.
 * @throws {Error} If the key is not a valid modifier key.
 */
webdriver.ActionSequence.prototype.keyUp = function(key) {
  webdriver.ActionSequence.checkModifierKey_(key);
  return this.scheduleKeyboardAction_('keyUp', [key]);
};


/**
 * Simulates typing multiple keys. Each modifier key encountered in the
 * sequence will not be released until it is encountered again. All key events
 * will be targetted at the currently focused element.
 * @param {...(string|!webdriver.Key|!Array.<(string|!webdriver.Key)>)} var_args
 *     The keys to type.
 * @return {!webdriver.ActionSequence} A self reference.
 * @throws {Error} If the key is not a valid modifier key.
 */
webdriver.ActionSequence.prototype.sendKeys = function(var_args) {
  var keys = goog.array.flatten(goog.array.slice(arguments, 0));
  return this.scheduleKeyboardAction_('sendKeys', keys);
};
