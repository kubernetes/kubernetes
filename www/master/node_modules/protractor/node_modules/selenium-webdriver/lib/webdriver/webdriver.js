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
 * @fileoverview The heart of the WebDriver JavaScript API.
 */

goog.provide('webdriver.Alert');
goog.provide('webdriver.AlertPromise');
goog.provide('webdriver.UnhandledAlertError');
goog.provide('webdriver.WebDriver');
goog.provide('webdriver.WebElement');
goog.provide('webdriver.WebElementPromise');

goog.require('bot.Error');
goog.require('bot.ErrorCode');
goog.require('bot.response');
goog.require('goog.array');
goog.require('goog.object');
goog.require('webdriver.ActionSequence');
goog.require('webdriver.Command');
goog.require('webdriver.CommandName');
goog.require('webdriver.Key');
goog.require('webdriver.Locator');
goog.require('webdriver.Session');
goog.require('webdriver.logging');
goog.require('webdriver.promise');
goog.require('webdriver.until');


//////////////////////////////////////////////////////////////////////////////
//
//  webdriver.WebDriver
//
//////////////////////////////////////////////////////////////////////////////



/**
 * Creates a new WebDriver client, which provides control over a browser.
 *
 * Every WebDriver command returns a {@code webdriver.promise.Promise} that
 * represents the result of that command. Callbacks may be registered on this
 * object to manipulate the command result or catch an expected error. Any
 * commands scheduled with a callback are considered sub-commands and will
 * execute before the next command in the current frame. For example:
 * <pre><code>
 *   var message = [];
 *   driver.call(message.push, message, 'a').then(function() {
 *     driver.call(message.push, message, 'b');
 *   });
 *   driver.call(message.push, message, 'c');
 *   driver.call(function() {
 *     alert('message is abc? ' + (message.join('') == 'abc'));
 *   });
 * </code></pre>
 *
 * @param {!(webdriver.Session|webdriver.promise.Promise)} session Either a
 *     known session or a promise that will be resolved to a session.
 * @param {!webdriver.CommandExecutor} executor The executor to use when
 *     sending commands to the browser.
 * @param {webdriver.promise.ControlFlow=} opt_flow The flow to
 *     schedule commands through. Defaults to the active flow object.
 * @constructor
 */
webdriver.WebDriver = function(session, executor, opt_flow) {

  /** @private {!(webdriver.Session|webdriver.promise.Promise)} */
  this.session_ = session;

  /** @private {!webdriver.CommandExecutor} */
  this.executor_ = executor;

  /** @private {!webdriver.promise.ControlFlow} */
  this.flow_ = opt_flow || webdriver.promise.controlFlow();
};


/**
 * Creates a new WebDriver client for an existing session.
 * @param {!webdriver.CommandExecutor} executor Command executor to use when
 *     querying for session details.
 * @param {string} sessionId ID of the session to attach to.
 * @param {webdriver.promise.ControlFlow=} opt_flow The control flow all driver
 *     commands should execute under. Defaults to the
 *     {@link webdriver.promise.controlFlow() currently active}  control flow.
 * @return {!webdriver.WebDriver} A new client for the specified session.
 */
webdriver.WebDriver.attachToSession = function(executor, sessionId, opt_flow) {
  return webdriver.WebDriver.acquireSession_(executor,
      new webdriver.Command(webdriver.CommandName.DESCRIBE_SESSION).
          setParameter('sessionId', sessionId),
      'WebDriver.attachToSession()',
      opt_flow);
};


/**
 * Creates a new WebDriver session.
 * @param {!webdriver.CommandExecutor} executor The executor to create the new
 *     session with.
 * @param {!webdriver.Capabilities} desiredCapabilities The desired
 *     capabilities for the new session.
 * @param {webdriver.promise.ControlFlow=} opt_flow The control flow all driver
 *     commands should execute under, including the initial session creation.
 *     Defaults to the {@link webdriver.promise.controlFlow() currently active} 
 *     control flow.
 * @return {!webdriver.WebDriver} The driver for the newly created session.
 */
webdriver.WebDriver.createSession = function(
    executor, desiredCapabilities, opt_flow) {
  return webdriver.WebDriver.acquireSession_(executor,
      new webdriver.Command(webdriver.CommandName.NEW_SESSION).
          setParameter('desiredCapabilities', desiredCapabilities),
      'WebDriver.createSession()',
      opt_flow);
};


/**
 * Sends a command to the server that is expected to return the details for a
 * {@link webdriver.Session}. This may either be an existing session, or a
 * newly created one.
 * @param {!webdriver.CommandExecutor} executor Command executor to use when
 *     querying for session details.
 * @param {!webdriver.Command} command The command to send to fetch the session
 *     details.
 * @param {string} description A descriptive debug label for this action.
 * @param {webdriver.promise.ControlFlow=} opt_flow The control flow all driver
 *     commands should execute under. Defaults to the
 *     {@link webdriver.promise.controlFlow() currently active} control flow.
 * @return {!webdriver.WebDriver} A new WebDriver client for the session.
 * @private
 */
webdriver.WebDriver.acquireSession_ = function(
    executor, command, description, opt_flow) {
  var flow = opt_flow || webdriver.promise.controlFlow();
  var session = flow.execute(function() {
    return webdriver.WebDriver.executeCommand_(executor, command).
        then(function(response) {
          bot.response.checkResponse(response);
          return new webdriver.Session(response['sessionId'],
              response['value']);
        });
  }, description);
  return new webdriver.WebDriver(session, executor, flow);
};


/**
 * Converts an object to its JSON representation in the WebDriver wire protocol.
 * When converting values of type object, the following steps will be taken:
 * <ol>
 * <li>if the object is a WebElement, the return value will be the element's
 *     server ID</li>
 * <li>if the object provides a "toJSON" function, the return value of this
 *     function will be returned</li>
 * <li>otherwise, the value of each key will be recursively converted according
 *     to the rules above.</li>
 * </ol>
 *
 * @param {*} obj The object to convert.
 * @return {!webdriver.promise.Promise.<?>} A promise that will resolve to the
 *     input value's JSON representation.
 * @private
 * @see http://code.google.com/p/selenium/wiki/JsonWireProtocol
 */
webdriver.WebDriver.toWireValue_ = function(obj) {
  if (webdriver.promise.isPromise(obj)) {
    return obj.then(webdriver.WebDriver.toWireValue_);
  }
  return webdriver.promise.fulfilled(convertValue(obj));

  function convertValue(value) {
    switch (goog.typeOf(value)) {
      case 'array':
        return convertKeys(value, true);
      case 'object':
        if (value instanceof webdriver.WebElement) {
          return value.getId();
        }
        if (goog.isFunction(value.toJSON)) {
          return value.toJSON();
        }
        if (goog.isNumber(value.nodeType) && goog.isString(value.nodeName)) {
          throw new TypeError(
              'Invalid argument type: ' + value.nodeName +
              '(' + value.nodeType + ')');
        }
        return convertKeys(value, false);
      case 'function':
        return '' + value;
      case 'undefined':
        return null;
      default:
        return value;
    }
  }

  function convertKeys(obj, isArray) {
    var numKeys = isArray ? obj.length : goog.object.getCount(obj);
    var ret = isArray ? new Array(numKeys) : {};
    if (!numKeys) {
      return webdriver.promise.fulfilled(ret);
    }

    var numResolved = 0;
    var done = webdriver.promise.defer();

    // forEach will stop iteration at undefined, where we want to convert
    // these to null and keep iterating.
    var forEachKey = !isArray ? goog.object.forEach : function(arr, fn) {
      var n = arr.length;
      for (var i = 0; i < n; i++) {
        fn(arr[i], i);
      }
    };

    forEachKey(obj, function(value, key) {
      if (webdriver.promise.isPromise(value)) {
        value.then(webdriver.WebDriver.toWireValue_).
            then(setValue, done.reject);
      } else {
        webdriver.promise.asap(convertValue(value), setValue, done.reject);
      }

      function setValue(value) {
        ret[key] = value;
        maybeFulfill();
      }
    });

    return done.promise;

    function maybeFulfill() {
      if (++numResolved === numKeys) {
        done.fulfill(ret);
      }
    }
  }
};


/**
 * Converts a value from its JSON representation according to the WebDriver wire
 * protocol. Any JSON object containing a
 * {@code webdriver.WebElement.ELEMENT_KEY} key will be decoded to a
 * {@code webdriver.WebElement} object. All other values will be passed through
 * as is.
 * @param {!webdriver.WebDriver} driver The driver instance to use as the
 *     parent of any unwrapped {@code webdriver.WebElement} values.
 * @param {*} value The value to convert.
 * @return {*} The converted value.
 * @see http://code.google.com/p/selenium/wiki/JsonWireProtocol
 * @private
 */
webdriver.WebDriver.fromWireValue_ = function(driver, value) {
  if (goog.isArray(value)) {
    value = goog.array.map(/**@type {goog.array.ArrayLike}*/ (value),
        goog.partial(webdriver.WebDriver.fromWireValue_, driver));
  } else if (value && goog.isObject(value) && !goog.isFunction(value)) {
    if (webdriver.WebElement.ELEMENT_KEY in value) {
      value = new webdriver.WebElement(driver, value);
    } else {
      value = goog.object.map(/**@type {!Object}*/ (value),
          goog.partial(webdriver.WebDriver.fromWireValue_, driver));
    }
  }
  return value;
};


/**
 * Translates a command to its wire-protocol representation before passing it
 * to the given {@code executor} for execution.
 * @param {!webdriver.CommandExecutor} executor The executor to use.
 * @param {!webdriver.Command} command The command to execute.
 * @return {!webdriver.promise.Promise} A promise that will resolve with the
 *     command response.
 * @private
 */
webdriver.WebDriver.executeCommand_ = function(executor, command) {
  return webdriver.WebDriver.toWireValue_(command.getParameters()).
      then(function(parameters) {
        command.setParameters(parameters);
        return webdriver.promise.checkedNodeCall(
            goog.bind(executor.execute, executor, command));
      });
};


/**
 * @return {!webdriver.promise.ControlFlow} The control flow used by this
 *     instance.
 */
webdriver.WebDriver.prototype.controlFlow = function() {
  return this.flow_;
};


/**
 * Schedules a {@code webdriver.Command} to be executed by this driver's
 * {@code webdriver.CommandExecutor}.
 * @param {!webdriver.Command} command The command to schedule.
 * @param {string} description A description of the command for debugging.
 * @return {!webdriver.promise.Promise.<T>} A promise that will be resolved
 *     with the command result.
 * @template T
 */
webdriver.WebDriver.prototype.schedule = function(command, description) {
  var self = this;

  checkHasNotQuit();
  command.setParameter('sessionId', this.session_);

  // If any of the command parameters are rejected promises, those
  // rejections may be reported as unhandled before the control flow
  // attempts to execute the command. To ensure parameters errors
  // propagate through the command itself, we resolve all of the
  // command parameters now, but suppress any errors until the ControlFlow
  // actually executes the command. This addresses scenarios like catching
  // an element not found error in:
  //
  //     driver.findElement(By.id('foo')).click().thenCatch(function(e) {
  //       if (e.code === bot.ErrorCode.NO_SUCH_ELEMENT) {
  //         // Do something.
  //       }
  //     });
  var prepCommand = webdriver.WebDriver.toWireValue_(command.getParameters());
  prepCommand.thenCatch(goog.nullFunction);

  var flow = this.flow_;
  var executor = this.executor_;
  return flow.execute(function() {
    // A call to WebDriver.quit() may have been scheduled in the same event
    // loop as this |command|, which would prevent us from detecting that the
    // driver has quit above.  Therefore, we need to make another quick check.
    // We still check above so we can fail as early as possible.
    checkHasNotQuit();

    // Retrieve resolved command parameters; any previously suppressed errors
    // will now propagate up through the control flow as part of the command
    // execution.
    return prepCommand.then(function(parameters) {
      command.setParameters(parameters);
      return webdriver.promise.checkedNodeCall(
          goog.bind(executor.execute, executor, command));
    });
  }, description).then(function(response) {
    try {
      bot.response.checkResponse(response);
    } catch (ex) {
      var value = response['value'];
      if (ex.code === bot.ErrorCode.UNEXPECTED_ALERT_OPEN) {
        var text = value && value['alert'] ? value['alert']['text'] : '';
        throw new webdriver.UnhandledAlertError(ex.message, text,
            new webdriver.Alert(self, text));
      }
      throw ex;
    }
    return webdriver.WebDriver.fromWireValue_(self, response['value']);
  });

  function checkHasNotQuit() {
    if (!self.session_) {
      throw new Error('This driver instance does not have a valid session ID ' +
                      '(did you call WebDriver.quit()?) and may no longer be ' +
                      'used.');
    }
  }
};


// ----------------------------------------------------------------------------
// Client command functions:
// ----------------------------------------------------------------------------


/**
 * @return {!webdriver.promise.Promise.<!webdriver.Session>} A promise for this
 *     client's session.
 */
webdriver.WebDriver.prototype.getSession = function() {
  return webdriver.promise.when(this.session_);
};


/**
 * @return {!webdriver.promise.Promise.<!webdriver.Capabilities>} A promise
 *     that will resolve with the this instance's capabilities.
 */
webdriver.WebDriver.prototype.getCapabilities = function() {
  return webdriver.promise.when(this.session_, function(session) {
    return session.getCapabilities();
  });
};


/**
 * Schedules a command to quit the current session. After calling quit, this
 * instance will be invalidated and may no longer be used to issue commands
 * against the browser.
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when the command has completed.
 */
webdriver.WebDriver.prototype.quit = function() {
  var result = this.schedule(
      new webdriver.Command(webdriver.CommandName.QUIT),
      'WebDriver.quit()');
  // Delete our session ID when the quit command finishes; this will allow us to
  // throw an error when attemnpting to use a driver post-quit.
  return result.thenFinally(goog.bind(function() {
    delete this.session_;
  }, this));
};


/**
 * Creates a new action sequence using this driver. The sequence will not be
 * scheduled for execution until {@link webdriver.ActionSequence#perform} is
 * called. Example:
 * <pre><code>
 *   driver.actions().
 *       mouseDown(element1).
 *       mouseMove(element2).
 *       mouseUp().
 *       perform();
 * </code></pre>
 * @return {!webdriver.ActionSequence} A new action sequence for this instance.
 */
webdriver.WebDriver.prototype.actions = function() {
  return new webdriver.ActionSequence(this);
};


/**
 * Schedules a command to execute JavaScript in the context of the currently
 * selected frame or window. The script fragment will be executed as the body
 * of an anonymous function. If the script is provided as a function object,
 * that function will be converted to a string for injection into the target
 * window.
 *
 * Any arguments provided in addition to the script will be included as script
 * arguments and may be referenced using the {@code arguments} object.
 * Arguments may be a boolean, number, string, or {@code webdriver.WebElement}.
 * Arrays and objects may also be used as script arguments as long as each item
 * adheres to the types previously mentioned.
 *
 * The script may refer to any variables accessible from the current window.
 * Furthermore, the script will execute in the window's context, thus
 * {@code document} may be used to refer to the current document. Any local
 * variables will not be available once the script has finished executing,
 * though global variables will persist.
 *
 * If the script has a return value (i.e. if the script contains a return
 * statement), then the following steps will be taken for resolving this
 * functions return value:
 * <ul>
 * <li>For a HTML element, the value will resolve to a
 *     {@code webdriver.WebElement}</li>
 * <li>Null and undefined return values will resolve to null</li>
 * <li>Booleans, numbers, and strings will resolve as is</li>
 * <li>Functions will resolve to their string representation</li>
 * <li>For arrays and objects, each member item will be converted according to
 *     the rules above</li>
 * </ul>
 *
 * @param {!(string|Function)} script The script to execute.
 * @param {...*} var_args The arguments to pass to the script.
 * @return {!webdriver.promise.Promise.<T>} A promise that will resolve to the
 *    scripts return value.
 * @template T
 */
webdriver.WebDriver.prototype.executeScript = function(script, var_args) {
  if (goog.isFunction(script)) {
    script = 'return (' + script + ').apply(null, arguments);';
  }
  var args = arguments.length > 1 ? goog.array.slice(arguments, 1) : [];
  return this.schedule(
      new webdriver.Command(webdriver.CommandName.EXECUTE_SCRIPT).
          setParameter('script', script).
          setParameter('args', args),
      'WebDriver.executeScript()');
};


/**
 * Schedules a command to execute asynchronous JavaScript in the context of the
 * currently selected frame or window. The script fragment will be executed as
 * the body of an anonymous function. If the script is provided as a function
 * object, that function will be converted to a string for injection into the
 * target window.
 *
 * Any arguments provided in addition to the script will be included as script
 * arguments and may be referenced using the {@code arguments} object.
 * Arguments may be a boolean, number, string, or {@code webdriver.WebElement}.
 * Arrays and objects may also be used as script arguments as long as each item
 * adheres to the types previously mentioned.
 *
 * Unlike executing synchronous JavaScript with
 * {@code webdriver.WebDriver.prototype.executeScript}, scripts executed with
 * this function must explicitly signal they are finished by invoking the
 * provided callback. This callback will always be injected into the
 * executed function as the last argument, and thus may be referenced with
 * {@code arguments[arguments.length - 1]}. The following steps will be taken
 * for resolving this functions return value against the first argument to the
 * script's callback function:
 * <ul>
 * <li>For a HTML element, the value will resolve to a
 *     {@code webdriver.WebElement}</li>
 * <li>Null and undefined return values will resolve to null</li>
 * <li>Booleans, numbers, and strings will resolve as is</li>
 * <li>Functions will resolve to their string representation</li>
 * <li>For arrays and objects, each member item will be converted according to
 *     the rules above</li>
 * </ul>
 *
 * Example #1: Performing a sleep that is synchronized with the currently
 * selected window:
 * <code><pre>
 * var start = new Date().getTime();
 * driver.executeAsyncScript(
 *     'window.setTimeout(arguments[arguments.length - 1], 500);').
 *     then(function() {
 *       console.log('Elapsed time: ' + (new Date().getTime() - start) + ' ms');
 *     });
 * </pre></code>
 *
 * Example #2: Synchronizing a test with an AJAX application:
 * <code><pre>
 * var button = driver.findElement(By.id('compose-button'));
 * button.click();
 * driver.executeAsyncScript(
 *     'var callback = arguments[arguments.length - 1];' +
 *     'mailClient.getComposeWindowWidget().onload(callback);');
 * driver.switchTo().frame('composeWidget');
 * driver.findElement(By.id('to')).sendKeys('dog@example.com');
 * </pre></code>
 *
 * Example #3: Injecting a XMLHttpRequest and waiting for the result. In this
 * example, the inject script is specified with a function literal. When using
 * this format, the function is converted to a string for injection, so it
 * should not reference any symbols not defined in the scope of the page under
 * test.
 * <code><pre>
 * driver.executeAsyncScript(function() {
 *   var callback = arguments[arguments.length - 1];
 *   var xhr = new XMLHttpRequest();
 *   xhr.open("GET", "/resource/data.json", true);
 *   xhr.onreadystatechange = function() {
 *     if (xhr.readyState == 4) {
 *       callback(xhr.responseText);
 *     }
 *   }
 *   xhr.send('');
 * }).then(function(str) {
 *   console.log(JSON.parse(str)['food']);
 * });
 * </pre></code>
 *
 * @param {!(string|Function)} script The script to execute.
 * @param {...*} var_args The arguments to pass to the script.
 * @return {!webdriver.promise.Promise.<T>} A promise that will resolve to the
 *    scripts return value.
 * @template T
 */
webdriver.WebDriver.prototype.executeAsyncScript = function(script, var_args) {
  if (goog.isFunction(script)) {
    script = 'return (' + script + ').apply(null, arguments);';
  }
  return this.schedule(
      new webdriver.Command(webdriver.CommandName.EXECUTE_ASYNC_SCRIPT).
          setParameter('script', script).
          setParameter('args', goog.array.slice(arguments, 1)),
      'WebDriver.executeScript()');
};


/**
 * Schedules a command to execute a custom function.
 * @param {function(...): (T|webdriver.promise.Promise.<T>)} fn The function to
 *     execute.
 * @param {Object=} opt_scope The object in whose scope to execute the function.
 * @param {...*} var_args Any arguments to pass to the function.
 * @return {!webdriver.promise.Promise.<T>} A promise that will be resolved'
 *     with the function's result.
 * @template T
 */
webdriver.WebDriver.prototype.call = function(fn, opt_scope, var_args) {
  var args = goog.array.slice(arguments, 2);
  var flow = this.flow_;
  return flow.execute(function() {
    return webdriver.promise.fullyResolved(args).then(function(args) {
      if (webdriver.promise.isGenerator(fn)) {
        args.unshift(fn, opt_scope);
        return webdriver.promise.consume.apply(null, args);
      }
      return fn.apply(opt_scope, args);
    });
  }, 'WebDriver.call(' + (fn.name || 'function') + ')');
};


/**
 * Schedules a command to wait for a condition to hold, as defined by some
 * user supplied function. If any errors occur while evaluating the wait, they
 * will be allowed to propagate.
 *
 * <p>In the event a condition returns a {@link webdriver.promise.Promise}, the
 * polling loop will wait for it to be resolved and use the resolved value for
 * evaluating whether the condition has been satisfied. The resolution time for
 * a promise is factored into whether a wait has timed out.
 *
 * @param {!(webdriver.until.Condition.<T>|
 *           function(!webdriver.WebDriver): T)} condition Either a condition
 *     object, or a function to evaluate as a condition.
 * @param {number} timeout How long to wait for the condition to be true.
 * @param {string=} opt_message An optional message to use if the wait times
 *     out.
 * @return {!webdriver.promise.Promise.<T>} A promise that will be fulfilled
 *     with the first truthy value returned by the condition function, or
 *     rejected if the condition times out.
 * @template T
 */
webdriver.WebDriver.prototype.wait = function(
    condition, timeout, opt_message) {
  var message = opt_message;
  var fn = /** @type {!Function} */(condition);
  if (condition instanceof webdriver.until.Condition) {
    message = message || condition.description();
    fn = condition.fn;
  }

  var driver = this;
  return this.flow_.wait(function() {
    if (webdriver.promise.isGenerator(fn)) {
      return webdriver.promise.consume(fn, null, [driver]);
    }
    return fn(driver);
  }, timeout, message);
};


/**
 * Schedules a command to make the driver sleep for the given amount of time.
 * @param {number} ms The amount of time, in milliseconds, to sleep.
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when the sleep has finished.
 */
webdriver.WebDriver.prototype.sleep = function(ms) {
  return this.flow_.timeout(ms, 'WebDriver.sleep(' + ms + ')');
};


/**
 * Schedules a command to retrieve they current window handle.
 * @return {!webdriver.promise.Promise.<string>} A promise that will be
 *     resolved with the current window handle.
 */
webdriver.WebDriver.prototype.getWindowHandle = function() {
  return this.schedule(
      new webdriver.Command(webdriver.CommandName.GET_CURRENT_WINDOW_HANDLE),
      'WebDriver.getWindowHandle()');
};


/**
 * Schedules a command to retrieve the current list of available window handles.
 * @return {!webdriver.promise.Promise.<!Array.<string>>} A promise that will
 *     be resolved with an array of window handles.
 */
webdriver.WebDriver.prototype.getAllWindowHandles = function() {
  return this.schedule(
      new webdriver.Command(webdriver.CommandName.GET_WINDOW_HANDLES),
      'WebDriver.getAllWindowHandles()');
};


/**
 * Schedules a command to retrieve the current page's source. The page source
 * returned is a representation of the underlying DOM: do not expect it to be
 * formatted or escaped in the same way as the response sent from the web
 * server.
 * @return {!webdriver.promise.Promise.<string>} A promise that will be
 *     resolved with the current page source.
 */
webdriver.WebDriver.prototype.getPageSource = function() {
  return this.schedule(
      new webdriver.Command(webdriver.CommandName.GET_PAGE_SOURCE),
      'WebDriver.getAllWindowHandles()');
};


/**
 * Schedules a command to close the current window.
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when this command has completed.
 */
webdriver.WebDriver.prototype.close = function() {
  return this.schedule(new webdriver.Command(webdriver.CommandName.CLOSE),
                       'WebDriver.close()');
};


/**
 * Schedules a command to navigate to the given URL.
 * @param {string} url The fully qualified URL to open.
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when the document has finished loading.
 */
webdriver.WebDriver.prototype.get = function(url) {
  return this.navigate().to(url);
};


/**
 * Schedules a command to retrieve the URL of the current page.
 * @return {!webdriver.promise.Promise.<string>} A promise that will be
 *     resolved with the current URL.
 */
webdriver.WebDriver.prototype.getCurrentUrl = function() {
  return this.schedule(
      new webdriver.Command(webdriver.CommandName.GET_CURRENT_URL),
      'WebDriver.getCurrentUrl()');
};


/**
 * Schedules a command to retrieve the current page's title.
 * @return {!webdriver.promise.Promise.<string>} A promise that will be
 *     resolved with the current page's title.
 */
webdriver.WebDriver.prototype.getTitle = function() {
  return this.schedule(new webdriver.Command(webdriver.CommandName.GET_TITLE),
                       'WebDriver.getTitle()');
};


/**
 * Schedule a command to find an element on the page. If the element cannot be
 * found, a {@link bot.ErrorCode.NO_SUCH_ELEMENT} result will be returned
 * by the driver. Unlike other commands, this error cannot be suppressed. In
 * other words, scheduling a command to find an element doubles as an assert
 * that the element is present on the page. To test whether an element is
 * present on the page, use {@link #isElementPresent} instead.
 *
 * <p>The search criteria for an element may be defined using one of the
 * factories in the {@link webdriver.By} namespace, or as a short-hand
 * {@link webdriver.By.Hash} object. For example, the following two statements
 * are equivalent:
 * <code><pre>
 * var e1 = driver.findElement(By.id('foo'));
 * var e2 = driver.findElement({id:'foo'});
 * </pre></code>
 *
 * <p>You may also provide a custom locator function, which takes as input
 * this WebDriver instance and returns a {@link webdriver.WebElement}, or a
 * promise that will resolve to a WebElement. For example, to find the first
 * visible link on a page, you could write:
 * <code><pre>
 * var link = driver.findElement(firstVisibleLink);
 *
 * function firstVisibleLink(driver) {
 *   var links = driver.findElements(By.tagName('a'));
 *   return webdriver.promise.filter(links, function(link) {
 *     return links.isDisplayed();
 *   }).then(function(visibleLinks) {
 *     return visibleLinks[0];
 *   });
 * }
 * </pre></code>
 *
 * <p>When running in the browser, a WebDriver cannot manipulate DOM elements
 * directly; it may do so only through a {@link webdriver.WebElement} reference.
 * This function may be used to generate a WebElement from a DOM element. A
 * reference to the DOM element will be stored in a known location and this
 * driver will attempt to retrieve it through {@link #executeScript}. If the
 * element cannot be found (eg, it belongs to a different document than the
 * one this instance is currently focused on), a
 * {@link bot.ErrorCode.NO_SUCH_ELEMENT} error will be returned.
 *
 * @param {!(webdriver.Locator|webdriver.By.Hash|Element|Function)} locator The
 *     locator to use.
 * @return {!webdriver.WebElement} A WebElement that can be used to issue
 *     commands against the located element. If the element is not found, the
 *     element will be invalidated and all scheduled commands aborted.
 */
webdriver.WebDriver.prototype.findElement = function(locator) {
  var id;
  if ('nodeType' in locator && 'ownerDocument' in locator) {
    var element = /** @type {!Element} */ (locator);
    id = this.findDomElement_(element).then(function(element) {
      if (!element) {
        throw new bot.Error(bot.ErrorCode.NO_SUCH_ELEMENT,
            'Unable to locate element. Is WebDriver focused on its ' +
                'ownerDocument\'s frame?');
      }
      return element;
    });
  } else {
    locator = webdriver.Locator.checkLocator(locator);
    if (goog.isFunction(locator)) {
      id = this.findElementInternal_(locator, this);
    } else {
      var command = new webdriver.Command(webdriver.CommandName.FIND_ELEMENT).
          setParameter('using', locator.using).
          setParameter('value', locator.value);
      id = this.schedule(command, 'WebDriver.findElement(' + locator + ')');
    }
  }
  return new webdriver.WebElementPromise(this, id);
};


/**
 * @param {!Function} locatorFn The locator function to use.
 * @param {!(webdriver.WebDriver|webdriver.WebElement)} context The search
 *     context.
 * @return {!webdriver.promise.Promise.<!webdriver.WebElement>} A
 *     promise that will resolve to a list of WebElements.
 * @private
 */
webdriver.WebDriver.prototype.findElementInternal_ = function(
    locatorFn, context) {
  return this.call(goog.partial(locatorFn, context)).then(function(result) {
    if (goog.isArray(result)) {
      result = result[0];
    }
    if (!(result instanceof webdriver.WebElement)) {
      throw new TypeError('Custom locator did not return a WebElement');
    }
    return result;
  });
};


/**
 * Locates a DOM element so that commands may be issued against it using the
 * {@link webdriver.WebElement} class. This is accomplished by storing a
 * reference to the element in an object on the element's ownerDocument.
 * {@link #executeScript} will then be used to create a WebElement from this
 * reference. This requires this driver to currently be focused on the
 * ownerDocument's window+frame.

 * @param {!Element} element The element to locate.
 * @return {!webdriver.promise.Promise.<webdriver.WebElement>} A promise that
 *     will be fulfilled with the located element, or null if the element
 *     could not be found.
 * @private
 */
webdriver.WebDriver.prototype.findDomElement_ = function(element) {
  var doc = element.ownerDocument;
  var store = doc['$webdriver$'] = doc['$webdriver$'] || {};
  var id = Math.floor(Math.random() * goog.now()).toString(36);
  store[id] = element;
  element[id] = id;

  function cleanUp() {
    delete store[id];
  }

  function lookupElement(id) {
    var store = document['$webdriver$'];
    if (!store) {
      return null;
    }

    var element = store[id];
    if (!element || element[id] !== id) {
      return null;
    }
    return element;
  }

  /** @type {!webdriver.promise.Promise.<webdriver.WebElement>} */
  var foundElement = this.executeScript(lookupElement, id);
  foundElement.thenFinally(cleanUp);
  return foundElement;
};


/**
 * Schedules a command to test if an element is present on the page.
 *
 * <p>If given a DOM element, this function will check if it belongs to the
 * document the driver is currently focused on. Otherwise, the function will
 * test if at least one element can be found with the given search criteria.
 *
 * @param {!(webdriver.Locator|webdriver.By.Hash|Element|
 *           Function)} locatorOrElement The locator to use, or the actual
 *     DOM element to be located by the server.
 * @return {!webdriver.promise.Promise.<boolean>} A promise that will resolve
 *     with whether the element is present on the page.
 */
webdriver.WebDriver.prototype.isElementPresent = function(locatorOrElement) {
  if ('nodeType' in locatorOrElement && 'ownerDocument' in locatorOrElement) {
    return this.findDomElement_(/** @type {!Element} */ (locatorOrElement)).
        then(function(result) { return !!result; });
  } else {
    return this.findElements.apply(this, arguments).then(function(result) {
      return !!result.length;
    });
  }
};


/**
 * Schedule a command to search for multiple elements on the page.
 *
 * @param {!(webdriver.Locator|webdriver.By.Hash|Function)} locator The locator
 *     strategy to use when searching for the element.
 * @return {!webdriver.promise.Promise.<!Array.<!webdriver.WebElement>>} A
 *     promise that will resolve to an array of WebElements.
 */
webdriver.WebDriver.prototype.findElements = function(locator) {
  locator = webdriver.Locator.checkLocator(locator);
  if (goog.isFunction(locator)) {
    return this.findElementsInternal_(locator, this);
  } else {
    var command = new webdriver.Command(webdriver.CommandName.FIND_ELEMENTS).
        setParameter('using', locator.using).
        setParameter('value', locator.value);
    return this.schedule(command, 'WebDriver.findElements(' + locator + ')');
  }
};


/**
 * @param {!Function} locatorFn The locator function to use.
 * @param {!(webdriver.WebDriver|webdriver.WebElement)} context The search
 *     context.
 * @return {!webdriver.promise.Promise.<!Array.<!webdriver.WebElement>>} A
 *     promise that will resolve to an array of WebElements.
 * @private
 */
webdriver.WebDriver.prototype.findElementsInternal_ = function(
    locatorFn, context) {
  return this.call(goog.partial(locatorFn, context)).then(function(result) {
    if (result instanceof webdriver.WebElement) {
      return [result];
    }

    if (!goog.isArray(result)) {
      return [];
    }

    return goog.array.filter(result, function(item) {
      return item instanceof webdriver.WebElement;
    });
  });
};


/**
 * Schedule a command to take a screenshot. The driver makes a best effort to
 * return a screenshot of the following, in order of preference:
 * <ol>
 *   <li>Entire page
 *   <li>Current window
 *   <li>Visible portion of the current frame
 *   <li>The screenshot of the entire display containing the browser
 * </ol>
 *
 * @return {!webdriver.promise.Promise.<string>} A promise that will be
 *     resolved to the screenshot as a base-64 encoded PNG.
 */
webdriver.WebDriver.prototype.takeScreenshot = function() {
  return this.schedule(new webdriver.Command(webdriver.CommandName.SCREENSHOT),
      'WebDriver.takeScreenshot()');
};


/**
 * @return {!webdriver.WebDriver.Options} The options interface for this
 *     instance.
 */
webdriver.WebDriver.prototype.manage = function() {
  return new webdriver.WebDriver.Options(this);
};


/**
 * @return {!webdriver.WebDriver.Navigation} The navigation interface for this
 *     instance.
 */
webdriver.WebDriver.prototype.navigate = function() {
  return new webdriver.WebDriver.Navigation(this);
};


/**
 * @return {!webdriver.WebDriver.TargetLocator} The target locator interface for
 *     this instance.
 */
webdriver.WebDriver.prototype.switchTo = function() {
  return new webdriver.WebDriver.TargetLocator(this);
};



/**
 * Interface for navigating back and forth in the browser history.
 * @param {!webdriver.WebDriver} driver The parent driver.
 * @constructor
 */
webdriver.WebDriver.Navigation = function(driver) {

  /** @private {!webdriver.WebDriver} */
  this.driver_ = driver;
};


/**
 * Schedules a command to navigate to a new URL.
 * @param {string} url The URL to navigate to.
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when the URL has been loaded.
 */
webdriver.WebDriver.Navigation.prototype.to = function(url) {
  return this.driver_.schedule(
      new webdriver.Command(webdriver.CommandName.GET).
          setParameter('url', url),
      'WebDriver.navigate().to(' + url + ')');
};


/**
 * Schedules a command to move backwards in the browser history.
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when the navigation event has completed.
 */
webdriver.WebDriver.Navigation.prototype.back = function() {
  return this.driver_.schedule(
      new webdriver.Command(webdriver.CommandName.GO_BACK),
      'WebDriver.navigate().back()');
};


/**
 * Schedules a command to move forwards in the browser history.
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when the navigation event has completed.
 */
webdriver.WebDriver.Navigation.prototype.forward = function() {
  return this.driver_.schedule(
      new webdriver.Command(webdriver.CommandName.GO_FORWARD),
      'WebDriver.navigate().forward()');
};


/**
 * Schedules a command to refresh the current page.
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when the navigation event has completed.
 */
webdriver.WebDriver.Navigation.prototype.refresh = function() {
  return this.driver_.schedule(
      new webdriver.Command(webdriver.CommandName.REFRESH),
      'WebDriver.navigate().refresh()');
};



/**
 * Provides methods for managing browser and driver state.
 * @param {!webdriver.WebDriver} driver The parent driver.
 * @constructor
 */
webdriver.WebDriver.Options = function(driver) {

  /** @private {!webdriver.WebDriver} */
  this.driver_ = driver;
};


/**
 * A JSON description of a browser cookie.
 * @typedef {{
 *     name: string,
 *     value: string,
 *     path: (string|undefined),
 *     domain: (string|undefined),
 *     secure: (boolean|undefined),
 *     expiry: (number|undefined)
 * }}
 * @see http://code.google.com/p/selenium/wiki/JsonWireProtocol#Cookie_JSON_Object
 */
webdriver.WebDriver.Options.Cookie;


/**
 * Schedules a command to add a cookie.
 * @param {string} name The cookie name.
 * @param {string} value The cookie value.
 * @param {string=} opt_path The cookie path.
 * @param {string=} opt_domain The cookie domain.
 * @param {boolean=} opt_isSecure Whether the cookie is secure.
 * @param {(number|!Date)=} opt_expiry When the cookie expires. If specified as
 *     a number, should be in milliseconds since midnight, January 1, 1970 UTC.
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when the cookie has been added to the page.
 */
webdriver.WebDriver.Options.prototype.addCookie = function(
    name, value, opt_path, opt_domain, opt_isSecure, opt_expiry) {
  // We do not allow '=' or ';' in the name.
  if (/[;=]/.test(name)) {
    throw Error('Invalid cookie name "' + name + '"');
  }

  // We do not allow ';' in value.
  if (/;/.test(value)) {
    throw Error('Invalid cookie value "' + value + '"');
  }

  var cookieString = name + '=' + value +
      (opt_domain ? ';domain=' + opt_domain : '') +
      (opt_path ? ';path=' + opt_path : '') +
      (opt_isSecure ? ';secure' : '');

  var expiry;
  if (goog.isDef(opt_expiry)) {
    var expiryDate;
    if (goog.isNumber(opt_expiry)) {
      expiryDate = new Date(opt_expiry);
    } else {
      expiryDate = /** @type {!Date} */ (opt_expiry);
      opt_expiry = expiryDate.getTime();
    }
    cookieString += ';expires=' + expiryDate.toUTCString();
    // Convert from milliseconds to seconds.
    expiry = Math.floor(/** @type {number} */ (opt_expiry) / 1000);
  }

  return this.driver_.schedule(
      new webdriver.Command(webdriver.CommandName.ADD_COOKIE).
          setParameter('cookie', {
            'name': name,
            'value': value,
            'path': opt_path,
            'domain': opt_domain,
            'secure': !!opt_isSecure,
            'expiry': expiry
          }),
      'WebDriver.manage().addCookie(' + cookieString + ')');
};


/**
 * Schedules a command to delete all cookies visible to the current page.
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when all cookies have been deleted.
 */
webdriver.WebDriver.Options.prototype.deleteAllCookies = function() {
  return this.driver_.schedule(
      new webdriver.Command(webdriver.CommandName.DELETE_ALL_COOKIES),
      'WebDriver.manage().deleteAllCookies()');
};


/**
 * Schedules a command to delete the cookie with the given name. This command is
 * a no-op if there is no cookie with the given name visible to the current
 * page.
 * @param {string} name The name of the cookie to delete.
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when the cookie has been deleted.
 */
webdriver.WebDriver.Options.prototype.deleteCookie = function(name) {
  return this.driver_.schedule(
      new webdriver.Command(webdriver.CommandName.DELETE_COOKIE).
          setParameter('name', name),
      'WebDriver.manage().deleteCookie(' + name + ')');
};


/**
 * Schedules a command to retrieve all cookies visible to the current page.
 * Each cookie will be returned as a JSON object as described by the WebDriver
 * wire protocol.
 * @return {!webdriver.promise.Promise.<
 *     !Array.<webdriver.WebDriver.Options.Cookie>>} A promise that will be
 *     resolved with the cookies visible to the current page.
 * @see http://code.google.com/p/selenium/wiki/JsonWireProtocol#Cookie_JSON_Object
 */
webdriver.WebDriver.Options.prototype.getCookies = function() {
  return this.driver_.schedule(
      new webdriver.Command(webdriver.CommandName.GET_ALL_COOKIES),
      'WebDriver.manage().getCookies()');
};


/**
 * Schedules a command to retrieve the cookie with the given name. Returns null
 * if there is no such cookie. The cookie will be returned as a JSON object as
 * described by the WebDriver wire protocol.
 * @param {string} name The name of the cookie to retrieve.
 * @return {!webdriver.promise.Promise.<?webdriver.WebDriver.Options.Cookie>} A
 *     promise that will be resolved with the named cookie, or {@code null}
 *     if there is no such cookie.
 * @see http://code.google.com/p/selenium/wiki/JsonWireProtocol#Cookie_JSON_Object
 */
webdriver.WebDriver.Options.prototype.getCookie = function(name) {
  return this.getCookies().then(function(cookies) {
    return goog.array.find(cookies, function(cookie) {
      return cookie && cookie['name'] == name;
    });
  });
};


/**
 * @return {!webdriver.WebDriver.Logs} The interface for managing driver
 *     logs.
 */
webdriver.WebDriver.Options.prototype.logs = function() {
  return new webdriver.WebDriver.Logs(this.driver_);
};


/**
 * @return {!webdriver.WebDriver.Timeouts} The interface for managing driver
 *     timeouts.
 */
webdriver.WebDriver.Options.prototype.timeouts = function() {
  return new webdriver.WebDriver.Timeouts(this.driver_);
};


/**
 * @return {!webdriver.WebDriver.Window} The interface for managing the
 *     current window.
 */
webdriver.WebDriver.Options.prototype.window = function() {
  return new webdriver.WebDriver.Window(this.driver_);
};



/**
 * An interface for managing timeout behavior for WebDriver instances.
 * @param {!webdriver.WebDriver} driver The parent driver.
 * @constructor
 */
webdriver.WebDriver.Timeouts = function(driver) {

  /** @private {!webdriver.WebDriver} */
  this.driver_ = driver;
};


/**
 * Specifies the amount of time the driver should wait when searching for an
 * element if it is not immediately present.
 * <p/>
 * When searching for a single element, the driver should poll the page
 * until the element has been found, or this timeout expires before failing
 * with a {@code bot.ErrorCode.NO_SUCH_ELEMENT} error. When searching
 * for multiple elements, the driver should poll the page until at least one
 * element has been found or this timeout has expired.
 * <p/>
 * Setting the wait timeout to 0 (its default value), disables implicit
 * waiting.
 * <p/>
 * Increasing the implicit wait timeout should be used judiciously as it
 * will have an adverse effect on test run time, especially when used with
 * slower location strategies like XPath.
 *
 * @param {number} ms The amount of time to wait, in milliseconds.
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when the implicit wait timeout has been set.
 */
webdriver.WebDriver.Timeouts.prototype.implicitlyWait = function(ms) {
  return this.driver_.schedule(
      new webdriver.Command(webdriver.CommandName.IMPLICITLY_WAIT).
          setParameter('ms', ms < 0 ? 0 : ms),
      'WebDriver.manage().timeouts().implicitlyWait(' + ms + ')');
};


/**
 * Sets the amount of time to wait, in milliseconds, for an asynchronous script
 * to finish execution before returning an error. If the timeout is less than or
 * equal to 0, the script will be allowed to run indefinitely.
 *
 * @param {number} ms The amount of time to wait, in milliseconds.
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when the script timeout has been set.
 */
webdriver.WebDriver.Timeouts.prototype.setScriptTimeout = function(ms) {
  return this.driver_.schedule(
      new webdriver.Command(webdriver.CommandName.SET_SCRIPT_TIMEOUT).
          setParameter('ms', ms < 0 ? 0 : ms),
      'WebDriver.manage().timeouts().setScriptTimeout(' + ms + ')');
};


/**
 * Sets the amount of time to wait for a page load to complete before returning
 * an error.  If the timeout is negative, page loads may be indefinite.
 * @param {number} ms The amount of time to wait, in milliseconds.
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when the timeout has been set.
 */
webdriver.WebDriver.Timeouts.prototype.pageLoadTimeout = function(ms) {
  return this.driver_.schedule(
      new webdriver.Command(webdriver.CommandName.SET_TIMEOUT).
          setParameter('type', 'page load').
          setParameter('ms', ms),
      'WebDriver.manage().timeouts().pageLoadTimeout(' + ms + ')');
};



/**
 * An interface for managing the current window.
 * @param {!webdriver.WebDriver} driver The parent driver.
 * @constructor
 */
webdriver.WebDriver.Window = function(driver) {

  /** @private {!webdriver.WebDriver} */
  this.driver_ = driver;
};


/**
 * Retrieves the window's current position, relative to the top left corner of
 * the screen.
 * @return {!webdriver.promise.Promise.<{x: number, y: number}>} A promise that
 *     will be resolved with the window's position in the form of a
 *     {x:number, y:number} object literal.
 */
webdriver.WebDriver.Window.prototype.getPosition = function() {
  return this.driver_.schedule(
      new webdriver.Command(webdriver.CommandName.GET_WINDOW_POSITION).
          setParameter('windowHandle', 'current'),
      'WebDriver.manage().window().getPosition()');
};


/**
 * Repositions the current window.
 * @param {number} x The desired horizontal position, relative to the left side
 *     of the screen.
 * @param {number} y The desired vertical position, relative to the top of the
 *     of the screen.
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when the command has completed.
 */
webdriver.WebDriver.Window.prototype.setPosition = function(x, y) {
  return this.driver_.schedule(
      new webdriver.Command(webdriver.CommandName.SET_WINDOW_POSITION).
          setParameter('windowHandle', 'current').
          setParameter('x', x).
          setParameter('y', y),
      'WebDriver.manage().window().setPosition(' + x + ', ' + y + ')');
};


/**
 * Retrieves the window's current size.
 * @return {!webdriver.promise.Promise.<{width: number, height: number}>} A
 *     promise that will be resolved with the window's size in the form of a
 *     {width:number, height:number} object literal.
 */
webdriver.WebDriver.Window.prototype.getSize = function() {
  return this.driver_.schedule(
      new webdriver.Command(webdriver.CommandName.GET_WINDOW_SIZE).
          setParameter('windowHandle', 'current'),
      'WebDriver.manage().window().getSize()');
};


/**
 * Resizes the current window.
 * @param {number} width The desired window width.
 * @param {number} height The desired window height.
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when the command has completed.
 */
webdriver.WebDriver.Window.prototype.setSize = function(width, height) {
  return this.driver_.schedule(
      new webdriver.Command(webdriver.CommandName.SET_WINDOW_SIZE).
          setParameter('windowHandle', 'current').
          setParameter('width', width).
          setParameter('height', height),
      'WebDriver.manage().window().setSize(' + width + ', ' + height + ')');
};


/**
 * Maximizes the current window.
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when the command has completed.
 */
webdriver.WebDriver.Window.prototype.maximize = function() {
  return this.driver_.schedule(
      new webdriver.Command(webdriver.CommandName.MAXIMIZE_WINDOW).
          setParameter('windowHandle', 'current'),
      'WebDriver.manage().window().maximize()');
};


/**
 * Interface for managing WebDriver log records.
 * @param {!webdriver.WebDriver} driver The parent driver.
 * @constructor
 */
webdriver.WebDriver.Logs = function(driver) {

  /** @private {!webdriver.WebDriver} */
  this.driver_ = driver;
};


/**
 * Fetches available log entries for the given type.
 *
 * <p/>Note that log buffers are reset after each call, meaning that
 * available log entries correspond to those entries not yet returned for a
 * given log type. In practice, this means that this call will return the
 * available log entries since the last call, or from the start of the
 * session.
 *
 * @param {!webdriver.logging.Type} type The desired log type.
 * @return {!webdriver.promise.Promise.<!Array.<!webdriver.logging.Entry>>} A
 *   promise that will resolve to a list of log entries for the specified
 *   type.
 */
webdriver.WebDriver.Logs.prototype.get = function(type) {
  return this.driver_.schedule(
      new webdriver.Command(webdriver.CommandName.GET_LOG).
          setParameter('type', type),
      'WebDriver.manage().logs().get(' + type + ')').
      then(function(entries) {
        return goog.array.map(entries, function(entry) {
          if (!(entry instanceof webdriver.logging.Entry)) {
            return new webdriver.logging.Entry(
                entry['level'], entry['message'], entry['timestamp'],
                entry['type']);
          }
          return entry;
        });
      });
};


/**
 * Retrieves the log types available to this driver.
 * @return {!webdriver.promise.Promise.<!Array.<!webdriver.logging.Type>>} A
 *     promise that will resolve to a list of available log types.
 */
webdriver.WebDriver.Logs.prototype.getAvailableLogTypes = function() {
  return this.driver_.schedule(
      new webdriver.Command(webdriver.CommandName.GET_AVAILABLE_LOG_TYPES),
      'WebDriver.manage().logs().getAvailableLogTypes()');
};



/**
 * An interface for changing the focus of the driver to another frame or window.
 * @param {!webdriver.WebDriver} driver The parent driver.
 * @constructor
 */
webdriver.WebDriver.TargetLocator = function(driver) {

  /** @private {!webdriver.WebDriver} */
  this.driver_ = driver;
};


/**
 * Schedules a command retrieve the {@code document.activeElement} element on
 * the current document, or {@code document.body} if activeElement is not
 * available.
 * @return {!webdriver.WebElementPromise} The active element.
 */
webdriver.WebDriver.TargetLocator.prototype.activeElement = function() {
  var id = this.driver_.schedule(
      new webdriver.Command(webdriver.CommandName.GET_ACTIVE_ELEMENT),
      'WebDriver.switchTo().activeElement()');
  return new webdriver.WebElementPromise(this.driver_, id);
};


/**
 * Schedules a command to switch focus of all future commands to the first frame
 * on the page.
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when the driver has changed focus to the default content.
 */
webdriver.WebDriver.TargetLocator.prototype.defaultContent = function() {
  return this.driver_.schedule(
      new webdriver.Command(webdriver.CommandName.SWITCH_TO_FRAME).
          setParameter('id', null),
      'WebDriver.switchTo().defaultContent()');
};


/**
 * Schedules a command to switch the focus of all future commands to another
 * frame on the page.
 * <p/>
 * If the frame is specified by a number, the command will switch to the frame
 * by its (zero-based) index into the {@code window.frames} collection.
 * <p/>
 * If the frame is specified by a string, the command will select the frame by
 * its name or ID. To select sub-frames, simply separate the frame names/IDs by
 * dots. As an example, "main.child" will select the frame with the name "main"
 * and then its child "child".
 * <p/>
 * If the specified frame can not be found, the deferred result will errback
 * with a {@code bot.ErrorCode.NO_SUCH_FRAME} error.
 * @param {string|number} nameOrIndex The frame locator.
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when the driver has changed focus to the specified frame.
 */
webdriver.WebDriver.TargetLocator.prototype.frame = function(nameOrIndex) {
  return this.driver_.schedule(
      new webdriver.Command(webdriver.CommandName.SWITCH_TO_FRAME).
          setParameter('id', nameOrIndex),
      'WebDriver.switchTo().frame(' + nameOrIndex + ')');
};


/**
 * Schedules a command to switch the focus of all future commands to another
 * window. Windows may be specified by their {@code window.name} attribute or
 * by its handle (as returned by {@code webdriver.WebDriver#getWindowHandles}).
 * <p/>
 * If the specificed window can not be found, the deferred result will errback
 * with a {@code bot.ErrorCode.NO_SUCH_WINDOW} error.
 * @param {string} nameOrHandle The name or window handle of the window to
 *     switch focus to.
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when the driver has changed focus to the specified window.
 */
webdriver.WebDriver.TargetLocator.prototype.window = function(nameOrHandle) {
  return this.driver_.schedule(
      new webdriver.Command(webdriver.CommandName.SWITCH_TO_WINDOW).
          setParameter('name', nameOrHandle),
      'WebDriver.switchTo().window(' + nameOrHandle + ')');
};


/**
 * Schedules a command to change focus to the active alert dialog. This command
 * will return a {@link bot.ErrorCode.NO_SUCH_ALERT} error if an alert dialog
 * is not currently open.
 * @return {!webdriver.AlertPromise} The open alert.
 */
webdriver.WebDriver.TargetLocator.prototype.alert = function() {
  var text = this.driver_.schedule(
      new webdriver.Command(webdriver.CommandName.GET_ALERT_TEXT),
      'WebDriver.switchTo().alert()');
  var driver = this.driver_;
  return new webdriver.AlertPromise(driver, text.then(function(text) {
    return new webdriver.Alert(driver, text);
  }));
};


/**
 * Simulate pressing many keys at once in a "chord". Takes a sequence of
 * {@link webdriver.Key}s or strings, appends each of the values to a string,
 * and adds the chord termination key ({@link webdriver.Key.NULL}) and returns
 * the resultant string.
 *
 * Note: when the low-level webdriver key handlers see Keys.NULL, active
 * modifier keys (CTRL/ALT/SHIFT/etc) release via a keyup event.
 *
 * @param {...string} var_args The key sequence to concatenate.
 * @return {string} The null-terminated key sequence.
 * @see http://code.google.com/p/webdriver/issues/detail?id=79
 */
webdriver.Key.chord = function(var_args) {
  var sequence = goog.array.reduce(
      goog.array.slice(arguments, 0),
      function(str, key) {
        return str + key;
      }, '');
  sequence += webdriver.Key.NULL;
  return sequence;
};


//////////////////////////////////////////////////////////////////////////////
//
//  webdriver.WebElement
//
//////////////////////////////////////////////////////////////////////////////



/**
 * Represents a DOM element. WebElements can be found by searching from the
 * document root using a {@code webdriver.WebDriver} instance, or by searching
 * under another {@code webdriver.WebElement}:
 * <pre><code>
 *   driver.get('http://www.google.com');
 *   var searchForm = driver.findElement(By.tagName('form'));
 *   var searchBox = searchForm.findElement(By.name('q'));
 *   searchBox.sendKeys('webdriver');
 * </code></pre>
 *
 * The WebElement is implemented as a promise for compatibility with the promise
 * API. It will always resolve itself when its internal state has been fully
 * resolved and commands may be issued against the element. This can be used to
 * catch errors when an element cannot be located on the page:
 * <pre><code>
 *   driver.findElement(By.id('not-there')).then(function(element) {
 *     alert('Found an element that was not expected to be there!');
 *   }, function(error) {
 *     alert('The element was not found, as expected');
 *   });
 * </code></pre>
 *
 * @param {!webdriver.WebDriver} driver The parent WebDriver instance for this
 *     element.
 * @param {!(webdriver.promise.Promise.<webdriver.WebElement.Id>|
 *           webdriver.WebElement.Id)} id The server-assigned opaque ID for the
 *     underlying DOM element.
 * @constructor
 */
webdriver.WebElement = function(driver, id) {

  /** @private {!webdriver.WebDriver} */
  this.driver_ = driver;

  /** @private {!webdriver.promise.Promise.<webdriver.WebElement.Id>} */
  this.id_ = id instanceof webdriver.promise.Promise ?
      id : webdriver.promise.fulfilled(id);
};


/**
 * Wire protocol definition of a WebElement ID.
 * @typedef {{ELEMENT: string}}
 * @see http://code.google.com/p/selenium/wiki/JsonWireProtocol
 */
webdriver.WebElement.Id;


/**
 * The property key used in the wire protocol to indicate that a JSON object
 * contains the ID of a WebElement.
 * @type {string}
 * @const
 */
webdriver.WebElement.ELEMENT_KEY = 'ELEMENT';


/**
 * Compares to WebElements for equality.
 * @param {!webdriver.WebElement} a A WebElement.
 * @param {!webdriver.WebElement} b A WebElement.
 * @return {!webdriver.promise.Promise.<boolean>} A promise that will be
 *     resolved to whether the two WebElements are equal.
 */
webdriver.WebElement.equals = function(a, b) {
  if (a == b) {
    return webdriver.promise.fulfilled(true);
  }
  var ids = [a.getId(), b.getId()];
  return webdriver.promise.all(ids).then(function(ids) {
    // If the two element's have the same ID, they should be considered
    // equal. Otherwise, they may still be equivalent, but we'll need to
    // ask the server to check for us.
    if (ids[0][webdriver.WebElement.ELEMENT_KEY] ==
        ids[1][webdriver.WebElement.ELEMENT_KEY]) {
      return true;
    }

    var command = new webdriver.Command(webdriver.CommandName.ELEMENT_EQUALS);
    command.setParameter('id', ids[0]);
    command.setParameter('other', ids[1]);
    return a.driver_.schedule(command, 'webdriver.WebElement.equals()');
  });
};


/**
 * @return {!webdriver.WebDriver} The parent driver for this instance.
 */
webdriver.WebElement.prototype.getDriver = function() {
  return this.driver_;
};


/**
 * @return {!webdriver.promise.Promise.<webdriver.WebElement.Id>} A promise
 *     that resolves to this element's JSON representation as defined by the
 *     WebDriver wire protocol.
 * @see http://code.google.com/p/selenium/wiki/JsonWireProtocol
 */
webdriver.WebElement.prototype.getId = function() {
  return this.id_;
};


/**
 * Schedules a command that targets this element with the parent WebDriver
 * instance. Will ensure this element's ID is included in the command parameters
 * under the "id" key.
 * @param {!webdriver.Command} command The command to schedule.
 * @param {string} description A description of the command for debugging.
 * @return {!webdriver.promise.Promise.<T>} A promise that will be resolved
 *     with the command result.
 * @template T
 * @see webdriver.WebDriver.prototype.schedule
 * @private
 */
webdriver.WebElement.prototype.schedule_ = function(command, description) {
  command.setParameter('id', this.getId());
  return this.driver_.schedule(command, description);
};


/**
 * Schedule a command to find a descendant of this element. If the element
 * cannot be found, a {@code bot.ErrorCode.NO_SUCH_ELEMENT} result will
 * be returned by the driver. Unlike other commands, this error cannot be
 * suppressed. In other words, scheduling a command to find an element doubles
 * as an assert that the element is present on the page. To test whether an
 * element is present on the page, use {@code #isElementPresent} instead.
 *
 * <p>The search criteria for an element may be defined using one of the
 * factories in the {@link webdriver.By} namespace, or as a short-hand
 * {@link webdriver.By.Hash} object. For example, the following two statements
 * are equivalent:
 * <code><pre>
 * var e1 = element.findElement(By.id('foo'));
 * var e2 = element.findElement({id:'foo'});
 * </pre></code>
 *
 * <p>You may also provide a custom locator function, which takes as input
 * this WebDriver instance and returns a {@link webdriver.WebElement}, or a
 * promise that will resolve to a WebElement. For example, to find the first
 * visible link on a page, you could write:
 * <code><pre>
 * var link = element.findElement(firstVisibleLink);
 *
 * function firstVisibleLink(element) {
 *   var links = element.findElements(By.tagName('a'));
 *   return webdriver.promise.filter(links, function(link) {
 *     return links.isDisplayed();
 *   }).then(function(visibleLinks) {
 *     return visibleLinks[0];
 *   });
 * }
 * </pre></code>
 *
 * @param {!(webdriver.Locator|webdriver.By.Hash|Function)} locator The
 *     locator strategy to use when searching for the element.
 * @return {!webdriver.WebElement} A WebElement that can be used to issue
 *     commands against the located element. If the element is not found, the
 *     element will be invalidated and all scheduled commands aborted.
 */
webdriver.WebElement.prototype.findElement = function(locator) {
  locator = webdriver.Locator.checkLocator(locator);
  var id;
  if (goog.isFunction(locator)) {
    id = this.driver_.findElementInternal_(locator, this);
  } else {
    var command = new webdriver.Command(
        webdriver.CommandName.FIND_CHILD_ELEMENT).
        setParameter('using', locator.using).
        setParameter('value', locator.value);
    id = this.schedule_(command, 'WebElement.findElement(' + locator + ')');
  }
  return new webdriver.WebElementPromise(this.driver_, id);
};


/**
 * Schedules a command to test if there is at least one descendant of this
 * element that matches the given search criteria.
 *
 * @param {!(webdriver.Locator|webdriver.By.Hash|Function)} locator The
 *     locator strategy to use when searching for the element.
 * @return {!webdriver.promise.Promise.<boolean>} A promise that will be
 *     resolved with whether an element could be located on the page.
 */
webdriver.WebElement.prototype.isElementPresent = function(locator) {
  return this.findElements(locator).then(function(result) {
    return !!result.length;
  });
};


/**
 * Schedules a command to find all of the descendants of this element that
 * match the given search criteria.
 *
 * @param {!(webdriver.Locator|webdriver.By.Hash|Function)} locator The
 *     locator strategy to use when searching for the elements.
 * @return {!webdriver.promise.Promise.<!Array.<!webdriver.WebElement>>} A
 *     promise that will resolve to an array of WebElements.
 */
webdriver.WebElement.prototype.findElements = function(locator) {
  locator = webdriver.Locator.checkLocator(locator);
  if (goog.isFunction(locator)) {
    return this.driver_.findElementsInternal_(locator, this);
  } else {
    var command = new webdriver.Command(
        webdriver.CommandName.FIND_CHILD_ELEMENTS).
        setParameter('using', locator.using).
        setParameter('value', locator.value);
    return this.schedule_(command, 'WebElement.findElements(' + locator + ')');
  }
};


/**
 * Schedules a command to click on this element.
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when the click command has completed.
 */
webdriver.WebElement.prototype.click = function() {
  return this.schedule_(
      new webdriver.Command(webdriver.CommandName.CLICK_ELEMENT),
      'WebElement.click()');
};


/**
 * Schedules a command to type a sequence on the DOM element represented by this
 * instance.
 * <p/>
 * Modifier keys (SHIFT, CONTROL, ALT, META) are stateful; once a modifier is
 * processed in the keysequence, that key state is toggled until one of the
 * following occurs:
 * <ul>
 * <li>The modifier key is encountered again in the sequence. At this point the
 * state of the key is toggled (along with the appropriate keyup/down events).
 * </li>
 * <li>The {@code webdriver.Key.NULL} key is encountered in the sequence. When
 * this key is encountered, all modifier keys current in the down state are
 * released (with accompanying keyup events). The NULL key can be used to
 * simulate common keyboard shortcuts:
 * <code><pre>
 *     element.sendKeys("text was",
 *                      webdriver.Key.CONTROL, "a", webdriver.Key.NULL,
 *                      "now text is");
 *     // Alternatively:
 *     element.sendKeys("text was",
 *                      webdriver.Key.chord(webdriver.Key.CONTROL, "a"),
 *                      "now text is");
 * </pre></code></li>
 * <li>The end of the keysequence is encountered. When there are no more keys
 * to type, all depressed modifier keys are released (with accompanying keyup
 * events).
 * </li>
 * </ul>
 * <strong>Note:</strong> On browsers where native keyboard events are not yet
 * supported (e.g. Firefox on OS X), key events will be synthesized. Special
 * punctionation keys will be synthesized according to a standard QWERTY en-us
 * keyboard layout.
 *
 * @param {...string} var_args The sequence of keys to
 *     type. All arguments will be joined into a single sequence (var_args is
 *     permitted for convenience).
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when all keys have been typed.
 */
webdriver.WebElement.prototype.sendKeys = function(var_args) {
  // Coerce every argument to a string. This protects us from users that
  // ignore the jsdoc and give us a number (which ends up causing problems on
  // the server, which requires strings).
  var keys = webdriver.promise.fullyResolved(goog.array.slice(arguments, 0)).
      then(function(args) {
        return goog.array.map(goog.array.slice(args, 0), function(key) {
          return key + '';
        });
      });
  return this.schedule_(
      new webdriver.Command(webdriver.CommandName.SEND_KEYS_TO_ELEMENT).
          setParameter('value', keys),
      'WebElement.sendKeys(' + keys + ')');
};


/**
 * Schedules a command to query for the tag/node name of this element.
 * @return {!webdriver.promise.Promise.<string>} A promise that will be
 *     resolved with the element's tag name.
 */
webdriver.WebElement.prototype.getTagName = function() {
  return this.schedule_(
      new webdriver.Command(webdriver.CommandName.GET_ELEMENT_TAG_NAME),
      'WebElement.getTagName()');
};


/**
 * Schedules a command to query for the computed style of the element
 * represented by this instance. If the element inherits the named style from
 * its parent, the parent will be queried for its value.  Where possible, color
 * values will be converted to their hex representation (e.g. #00ff00 instead of
 * rgb(0, 255, 0)).
 * <p/>
 * <em>Warning:</em> the value returned will be as the browser interprets it, so
 * it may be tricky to form a proper assertion.
 *
 * @param {string} cssStyleProperty The name of the CSS style property to look
 *     up.
 * @return {!webdriver.promise.Promise.<string>} A promise that will be
 *     resolved with the requested CSS value.
 */
webdriver.WebElement.prototype.getCssValue = function(cssStyleProperty) {
  var name = webdriver.CommandName.GET_ELEMENT_VALUE_OF_CSS_PROPERTY;
  return this.schedule_(
      new webdriver.Command(name).
          setParameter('propertyName', cssStyleProperty),
      'WebElement.getCssValue(' + cssStyleProperty + ')');
};


/**
 * Schedules a command to query for the value of the given attribute of the
 * element. Will return the current value, even if it has been modified after
 * the page has been loaded. More exactly, this method will return the value of
 * the given attribute, unless that attribute is not present, in which case the
 * value of the property with the same name is returned. If neither value is
 * set, null is returned (for example, the "value" property of a textarea
 * element). The "style" attribute is converted as best can be to a
 * text representation with a trailing semi-colon. The following are deemed to
 * be "boolean" attributes and will return either "true" or null:
 *
 * <p>async, autofocus, autoplay, checked, compact, complete, controls, declare,
 * defaultchecked, defaultselected, defer, disabled, draggable, ended,
 * formnovalidate, hidden, indeterminate, iscontenteditable, ismap, itemscope,
 * loop, multiple, muted, nohref, noresize, noshade, novalidate, nowrap, open,
 * paused, pubdate, readonly, required, reversed, scoped, seamless, seeking,
 * selected, spellcheck, truespeed, willvalidate
 *
 * <p>Finally, the following commonly mis-capitalized attribute/property names
 * are evaluated as expected:
 * <ul>
 *   <li>"class"
 *   <li>"readonly"
 * </ul>
 * @param {string} attributeName The name of the attribute to query.
 * @return {!webdriver.promise.Promise.<?string>} A promise that will be
 *     resolved with the attribute's value. The returned value will always be
 *     either a string or null.
 */
webdriver.WebElement.prototype.getAttribute = function(attributeName) {
  return this.schedule_(
      new webdriver.Command(webdriver.CommandName.GET_ELEMENT_ATTRIBUTE).
          setParameter('name', attributeName),
      'WebElement.getAttribute(' + attributeName + ')');
};


/**
 * Get the visible (i.e. not hidden by CSS) innerText of this element, including
 * sub-elements, without any leading or trailing whitespace.
 * @return {!webdriver.promise.Promise.<string>} A promise that will be
 *     resolved with the element's visible text.
 */
webdriver.WebElement.prototype.getText = function() {
  return this.schedule_(
      new webdriver.Command(webdriver.CommandName.GET_ELEMENT_TEXT),
      'WebElement.getText()');
};


/**
 * Schedules a command to compute the size of this element's bounding box, in
 * pixels.
 * @return {!webdriver.promise.Promise.<{width: number, height: number}>} A
 *     promise that will be resolved with the element's size as a
 *     {@code {width:number, height:number}} object.
 */
webdriver.WebElement.prototype.getSize = function() {
  return this.schedule_(
      new webdriver.Command(webdriver.CommandName.GET_ELEMENT_SIZE),
      'WebElement.getSize()');
};


/**
 * Schedules a command to compute the location of this element in page space.
 * @return {!webdriver.promise.Promise.<{x: number, y: number}>} A promise that
 *     will be resolved to the element's location as a
 *     {@code {x:number, y:number}} object.
 */
webdriver.WebElement.prototype.getLocation = function() {
  return this.schedule_(
      new webdriver.Command(webdriver.CommandName.GET_ELEMENT_LOCATION),
      'WebElement.getLocation()');
};


/**
 * Schedules a command to query whether the DOM element represented by this
 * instance is enabled, as dicted by the {@code disabled} attribute.
 * @return {!webdriver.promise.Promise.<boolean>} A promise that will be
 *     resolved with whether this element is currently enabled.
 */
webdriver.WebElement.prototype.isEnabled = function() {
  return this.schedule_(
      new webdriver.Command(webdriver.CommandName.IS_ELEMENT_ENABLED),
      'WebElement.isEnabled()');
};


/**
 * Schedules a command to query whether this element is selected.
 * @return {!webdriver.promise.Promise.<boolean>} A promise that will be
 *     resolved with whether this element is currently selected.
 */
webdriver.WebElement.prototype.isSelected = function() {
  return this.schedule_(
      new webdriver.Command(webdriver.CommandName.IS_ELEMENT_SELECTED),
      'WebElement.isSelected()');
};


/**
 * Schedules a command to submit the form containing this element (or this
 * element if it is a FORM element). This command is a no-op if the element is
 * not contained in a form.
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when the form has been submitted.
 */
webdriver.WebElement.prototype.submit = function() {
  return this.schedule_(
      new webdriver.Command(webdriver.CommandName.SUBMIT_ELEMENT),
      'WebElement.submit()');
};


/**
 * Schedules a command to clear the {@code value} of this element. This command
 * has no effect if the underlying DOM element is neither a text INPUT element
 * nor a TEXTAREA element.
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when the element has been cleared.
 */
webdriver.WebElement.prototype.clear = function() {
  return this.schedule_(
      new webdriver.Command(webdriver.CommandName.CLEAR_ELEMENT),
      'WebElement.clear()');
};


/**
 * Schedules a command to test whether this element is currently displayed.
 * @return {!webdriver.promise.Promise.<boolean>} A promise that will be
 *     resolved with whether this element is currently visible on the page.
 */
webdriver.WebElement.prototype.isDisplayed = function() {
  return this.schedule_(
      new webdriver.Command(webdriver.CommandName.IS_ELEMENT_DISPLAYED),
      'WebElement.isDisplayed()');
};


/**
 * Schedules a command to retrieve the outer HTML of this element.
 * @return {!webdriver.promise.Promise.<string>} A promise that will be
 *     resolved with the element's outer HTML.
 */
webdriver.WebElement.prototype.getOuterHtml = function() {
  return this.driver_.executeScript(function() {
    var element = arguments[0];
    if ('outerHTML' in element) {
      return element.outerHTML;
    } else {
      var div = element.ownerDocument.createElement('div');
      div.appendChild(element.cloneNode(true));
      return div.innerHTML;
    }
  }, this);
};


/**
 * Schedules a command to retrieve the inner HTML of this element.
 * @return {!webdriver.promise.Promise.<string>} A promise that will be
 *     resolved with the element's inner HTML.
 */
webdriver.WebElement.prototype.getInnerHtml = function() {
  return this.driver_.executeScript('return arguments[0].innerHTML', this);
};



/**
 * WebElementPromise is a promise that will be fulfilled with a WebElement.
 * This serves as a forward proxy on WebElement, allowing calls to be
 * scheduled without directly on this instance before the underlying
 * WebElement has been fulfilled. In other words, the following two statements
 * are equivalent:
 * <pre><code>
 *     driver.findElement({id: 'my-button'}).click();
 *     driver.findElement({id: 'my-button'}).then(function(el) {
 *       return el.click();
 *     });
 * </code></pre>
 *
 * @param {!webdriver.WebDriver} driver The parent WebDriver instance for this
 *     element.
 * @param {!webdriver.promise.Promise.<!webdriver.WebElement>} el A promise
 *     that will resolve to the promised element.
 * @constructor
 * @extends {webdriver.WebElement}
 * @implements {webdriver.promise.Thenable.<!webdriver.WebElement>}
 * @final
 */
webdriver.WebElementPromise = function(driver, el) {
  webdriver.WebElement.call(this, driver, {'ELEMENT': 'unused'});

  /** @override */
  this.cancel = goog.bind(el.cancel, el);

  /** @override */
  this.isPending = goog.bind(el.isPending, el);

  /** @override */
  this.then = goog.bind(el.then, el);

  /** @override */
  this.thenCatch = goog.bind(el.thenCatch, el);

  /** @override */
  this.thenFinally = goog.bind(el.thenFinally, el);

  /**
   * Defers returning the element ID until the wrapped WebElement has been
   * resolved.
   * @override
   */
  this.getId = function() {
    return el.then(function(el) {
      return el.getId();
    });
  };
};
goog.inherits(webdriver.WebElementPromise, webdriver.WebElement);


/**
 * Represents a modal dialog such as {@code alert}, {@code confirm}, or
 * {@code prompt}. Provides functions to retrieve the message displayed with
 * the alert, accept or dismiss the alert, and set the response text (in the
 * case of {@code prompt}).
 * @param {!webdriver.WebDriver} driver The driver controlling the browser this
 *     alert is attached to.
 * @param {string} text The message text displayed with this alert.
 * @constructor
 */
webdriver.Alert = function(driver, text) {
  /** @private {!webdriver.WebDriver} */
  this.driver_ = driver;

  /** @private {!webdriver.promise.Promise.<string>} */
  this.text_ = webdriver.promise.when(text);
};


/**
 * Retrieves the message text displayed with this alert. For instance, if the
 * alert were opened with alert("hello"), then this would return "hello".
 * @return {!webdriver.promise.Promise.<string>} A promise that will be
 *     resolved to the text displayed with this alert.
 */
webdriver.Alert.prototype.getText = function() {
  return this.text_;
};


/**
 * Accepts this alert.
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when this command has completed.
 */
webdriver.Alert.prototype.accept = function() {
  return this.driver_.schedule(
      new webdriver.Command(webdriver.CommandName.ACCEPT_ALERT),
      'WebDriver.switchTo().alert().accept()');
};


/**
 * Dismisses this alert.
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when this command has completed.
 */
webdriver.Alert.prototype.dismiss = function() {
  return this.driver_.schedule(
      new webdriver.Command(webdriver.CommandName.DISMISS_ALERT),
      'WebDriver.switchTo().alert().dismiss()');
};


/**
 * Sets the response text on this alert. This command will return an error if
 * the underlying alert does not support response text (e.g. window.alert and
 * window.confirm).
 * @param {string} text The text to set.
 * @return {!webdriver.promise.Promise.<void>} A promise that will be resolved
 *     when this command has completed.
 */
webdriver.Alert.prototype.sendKeys = function(text) {
  return this.driver_.schedule(
      new webdriver.Command(webdriver.CommandName.SET_ALERT_TEXT).
          setParameter('text', text),
      'WebDriver.switchTo().alert().sendKeys(' + text + ')');
};



/**
 * AlertPromise is a promise that will be fulfilled with an Alert. This promise
 * serves as a forward proxy on an Alert, allowing calls to be scheduled
 * directly on this instance before the underlying Alert has been fulfilled. In
 * other words, the following two statements are equivalent:
 * <pre><code>
 *     driver.switchTo().alert().dismiss();
 *     driver.switchTo().alert().then(function(alert) {
 *       return alert.dismiss();
 *     });
 * </code></pre>
 *
 * @param {!webdriver.WebDriver} driver The driver controlling the browser this
 *     alert is attached to.
 * @param {!webdriver.promise.Thenable.<!webdriver.Alert>} alert A thenable
 *     that will be fulfilled with the promised alert.
 * @constructor
 * @extends {webdriver.Alert}
 * @implements {webdriver.promise.Thenable.<!webdriver.Alert>}
 * @final
 */
webdriver.AlertPromise = function(driver, alert) {
  webdriver.Alert.call(this, driver, 'unused');

  /** @override */
  this.cancel = goog.bind(alert.cancel, alert);

  /** @override */
  this.isPending = goog.bind(alert.isPending, alert);

  /** @override */
  this.then = goog.bind(alert.then, alert);

  /** @override */
  this.thenCatch = goog.bind(alert.thenCatch, alert);

  /** @override */
  this.thenFinally = goog.bind(alert.thenFinally, alert);

  /**
   * Defer returning text until the promised alert has been resolved.
   * @override
   */
  this.getText = function() {
    return alert.then(function(alert) {
      return alert.getText();
    });
  };

  /**
   * Defers action until the alert has been located.
   * @override
   */
  this.accept = function() {
    return alert.then(function(alert) {
      return alert.accept();
    });
  };

  /**
   * Defers action until the alert has been located.
   * @override
   */
  this.dismiss = function() {
    return alert.then(function(alert) {
      return alert.dismiss();
    });
  };

  /**
   * Defers action until the alert has been located.
   * @override
   */
  this.sendKeys = function(text) {
    return alert.then(function(alert) {
      return alert.sendKeys(text);
    });
  };
};
goog.inherits(webdriver.AlertPromise, webdriver.Alert);



/**
 * An error returned to indicate that there is an unhandled modal dialog on the
 * current page.
 * @param {string} message The error message.
 * @param {string} text The text displayed with the unhandled alert.
 * @param {!webdriver.Alert} alert The alert handle.
 * @constructor
 * @extends {bot.Error}
 */
webdriver.UnhandledAlertError = function(message, text, alert) {
  goog.base(this, bot.ErrorCode.UNEXPECTED_ALERT_OPEN, message);

  /** @private {string} */
  this.text_ = text;

  /** @private {!webdriver.Alert} */
  this.alert_ = alert;
};
goog.inherits(webdriver.UnhandledAlertError, bot.Error);


/**
 * @return {string} The text displayed with the unhandled alert.
 */
webdriver.UnhandledAlertError.prototype.getAlertText = function() {
  return this.text_;
};


/**
 * @return {!webdriver.Alert} The open alert.
 * @deprecated Use {@link #getAlertText}. This method will be removed in
 *     2.45.0.
 */
webdriver.UnhandledAlertError.prototype.getAlert = function() {
  return this.alert_;
};
