var url = require('url');
var webdriver = require('selenium-webdriver');
var helper = require('./util');
var log = require('./logger.js');
var ElementArrayFinder = require('./element').ElementArrayFinder;
var ElementFinder = require('./element').ElementFinder;
var build$ = require('./element').build$;
var build$$ = require('./element').build$$;

var clientSideScripts = require('./clientsidescripts.js');
var ProtractorBy = require('./locators.js').ProtractorBy;
var ExpectedConditions = require('./expectedConditions.js');

/* global angular */

var DEFER_LABEL = 'NG_DEFER_BOOTSTRAP!';
var DEFAULT_RESET_URL = 'data:text/html,<html></html>';
var DEFAULT_GET_PAGE_TIMEOUT = 10000;

/*
 * Mix in other webdriver functionality to be accessible via protractor.
 */
for (var foo in webdriver) {
  exports[foo] = webdriver[foo];
}

/**
 * @type {ProtractorBy}
 */
exports.By = new ProtractorBy();

/**
 * @type {ElementFinder}
 */
exports.ElementFinder = ElementFinder;

/**
 * @type {ElementArrayFinder}
 */
exports.ElementArrayFinder = ElementArrayFinder;

/**
 * @type {ExpectedConditions}
 */
exports.ExpectedConditions = new ExpectedConditions();

/**
 * Mix a function from one object onto another. The function will still be
 * called in the context of the original object.
 *
 * @private
 * @param {Object} to
 * @param {Object} from
 * @param {string} fnName
 * @param {function=} setupFn
 */
var mixin = function(to, from, fnName, setupFn) {
  to[fnName] = function() {
    if (setupFn) {
      setupFn();
    }
    return from[fnName].apply(from, arguments);
  };
};

/**
 * Build the helper 'element' function for a given instance of Protractor.
 *
 * @private
 * @param {Protractor} ptor
 * @return {function(webdriver.Locator): ElementFinder}
 */
var buildElementHelper = function(ptor) {
  var element = function(locator) {
    return new ElementArrayFinder(ptor).all(locator).toElementFinder_();
  };

  element.all = function(locator) {
    return new ElementArrayFinder(ptor).all(locator);
  };

  return element;
};

/**
 * @alias browser
 * @constructor
 * @extends {webdriver.WebDriver}
 * @param {webdriver.WebDriver} webdriver
 * @param {string=} opt_baseUrl A base URL to run get requests against.
 * @param {string=} opt_rootElement  Selector element that has an ng-app in
 *     scope.
 */
var Protractor = function(webdriverInstance, opt_baseUrl, opt_rootElement) {
  // These functions should delegate to the webdriver instance, but should
  // wait for Angular to sync up before performing the action. This does not
  // include functions which are overridden by protractor below.
  var methodsToSync = ['getCurrentUrl', 'getPageSource', 'getTitle'];

  // Mix all other driver functionality into Protractor.
  for (var method in webdriverInstance) {
    if (!this[method] && typeof webdriverInstance[method] == 'function') {
      if (methodsToSync.indexOf(method) !== -1) {
        mixin(this, webdriverInstance, method, this.waitForAngular.bind(this));
      } else {
        mixin(this, webdriverInstance, method);
      }
    }
  }
  var self = this;

  /**
   * The wrapped webdriver instance. Use this to interact with pages that do
   * not contain Angular (such as a log-in screen).
   *
   * @type {webdriver.WebDriver}
   */
  this.driver = webdriverInstance;

  /**
   * Helper function for finding elements.
   *
   * @type {function(webdriver.Locator): ElementFinder}
   */
  this.element = buildElementHelper(this);

  /**
   * Shorthand function for finding elements by css.
   *
   * @type {function(string): ElementFinder}
   */
  this.$ = build$(this.element, webdriver.By);

  /**
   * Shorthand function for finding arrays of elements by css.
   *
   * @type {function(string): ElementArrayFinder}
   */
  this.$$ = build$$(this.element, webdriver.By);

  /**
   * All get methods will be resolved against this base URL. Relative URLs are =
   * resolved the way anchor tags resolve.
   *
   * @type {string}
   */
  this.baseUrl = opt_baseUrl || '';

  /**
   * The css selector for an element on which to find Angular. This is usually
   * 'body' but if your ng-app is on a subsection of the page it may be
   * a subelement.
   *
   * @type {string}
   */
  this.rootEl = opt_rootElement || 'body';

  /**
   * If true, Protractor will not attempt to synchronize with the page before
   * performing actions. This can be harmful because Protractor will not wait
   * until $timeouts and $http calls have been processed, which can cause
   * tests to become flaky. This should be used only when necessary, such as
   * when a page continuously polls an API using $timeout.
   *
   * @type {boolean}
   */
  this.ignoreSynchronization = false;

  /**
   * Timeout in milliseconds to wait for pages to load when calling `get`.
   *
   * @type {number}
   */
  this.getPageTimeout = DEFAULT_GET_PAGE_TIMEOUT;

  /**
   * An object that holds custom test parameters.
   *
   * @type {Object}
   */
  this.params = {};

  /**
   * The reset URL to use between page loads.
   *
   * @type {string}
   */
  this.resetUrl = DEFAULT_RESET_URL;
  this.driver.getCapabilities().then(function(caps) {
    // Internet Explorer does not accept data URLs, which are the default
    // reset URL for Protractor.
    // Safari accepts data urls, but SafariDriver fails after one is used.
    var browserName = caps.get('browserName');
    if (browserName === 'internet explorer' || browserName === 'safari') {
      self.resetUrl = 'about:blank';
    }
  });

  /**
   * Information about mock modules that will be installed during every
   * get().
   *
   * @type {Array<{name: string, script: function|string, args: Array.<string>}>}
   */
  this.mockModules_ = [];

  this.addBaseMockModules_();
};

/**
 * The same as {@code webdriver.WebDriver.prototype.executeScript},
 * but with a customized description for debugging.
 *
 * @private
 * @param {!(string|Function)} script The script to execute.
 * @param {string} description A description of the command for debugging.
 * @param {...*} var_args The arguments to pass to the script.
 * @return {!webdriver.promise.Promise.<T>} A promise that will resolve to the
 *    scripts return value.
 * @template T
 */
Protractor.prototype.executeScript_ = function(script, description) {
  if (typeof script === 'function') {
    script = 'return (' + script + ').apply(null, arguments);';
  }

  return this.driver.schedule(
      new webdriver.Command(webdriver.CommandName.EXECUTE_SCRIPT).
          setParameter('script', script).
          setParameter('args', Array.prototype.slice.call(arguments, 2)),
      description);
};

/**
 * The same as {@code webdriver.WebDriver.prototype.executeAsyncScript},
 * but with a customized description for debugging.
 *
 * @private
 * @param {!(string|Function)} script The script to execute.
 * @param {string} description A description for debugging purposes.
 * @param {...*} var_args The arguments to pass to the script.
 * @return {!webdriver.promise.Promise.<T>} A promise that will resolve to the
 *    scripts return value.
 * @template T
 */
Protractor.prototype.executeAsyncScript_ =
    function(script, description) {
      if (typeof script === 'function') {
        script = 'return (' + script + ').apply(null, arguments);';
      }
      return this.driver.schedule(
          new webdriver.Command(webdriver.CommandName.EXECUTE_ASYNC_SCRIPT).
              setParameter('script', script).
              setParameter('args', Array.prototype.slice.call(arguments, 2)),
          description);
    };

/**
 * Instruct webdriver to wait until Angular has finished rendering and has
 * no outstanding $http or $timeout calls before continuing.
 * Note that Protractor automatically applies this command before every
 * WebDriver action.
 *
 * @param {string=} opt_description An optional description to be added
 *     to webdriver logs.
 * @return {!webdriver.promise.Promise} A promise that will resolve to the
 *    scripts return value.
 */
Protractor.prototype.waitForAngular = function(opt_description) {
  var description = opt_description ? ' - ' + opt_description : '';
  if (this.ignoreSynchronization) {
    return webdriver.promise.fulfilled();
  }
  return this.executeAsyncScript_(
      clientSideScripts.waitForAngular,
      'Protractor.waitForAngular()' + description,
      this.rootEl).
      then(function(browserErr) {
        if (browserErr) {
          throw 'Error while waiting for Protractor to ' +
                'sync with the page: ' + JSON.stringify(browserErr);
        }
    }).then(null, function(err) {
      var timeout;
      if (/asynchronous script timeout/.test(err.message)) {
        // Timeout on Chrome
        timeout = /-?[\d\.]*\ seconds/.exec(err.message);
      } else if (/Timed out waiting for async script/.test(err.message)) {
        // Timeout on Firefox
        timeout = /-?[\d\.]*ms/.exec(err.message);
      } else if (/Timed out waiting for an asynchronous script/.test(err.message)) {
        // Timeout on Safari
        timeout = /-?[\d\.]*\ ms/.exec(err.message);
      }
      if (timeout) {
        throw 'Timed out waiting for Protractor to synchronize with ' +
            'the page after ' + timeout + '. Please see ' +
            'https://github.com/angular/protractor/blob/master/docs/faq.md';
      } else {
        throw err;
      }
    });
};

/**
 * Waits for Angular to finish rendering before searching for elements.
 * @see webdriver.WebDriver.findElement
 * @return {!webdriver.WebElement}
 */
Protractor.prototype.findElement = function(locator) {
  return this.element(locator).getWebElement();
};

/**
 * Waits for Angular to finish rendering before searching for elements.
 * @see webdriver.WebDriver.findElements
 * @return {!webdriver.promise.Promise} A promise that will be resolved to an
 *     array of the located {@link webdriver.WebElement}s.
 */
Protractor.prototype.findElements = function(locator) {
  return this.element.all(locator).getWebElements();
};

/**
 * Tests if an element is present on the page.
 * @see webdriver.WebDriver.isElementPresent
 * @return {!webdriver.promise.Promise} A promise that will resolve to whether
 *     the element is present on the page.
 */
Protractor.prototype.isElementPresent = function(locatorOrElement) {
  var element = (locatorOrElement instanceof webdriver.promise.Promise) ?
      locatorOrElement : this.element(locatorOrElement);
  return element.isPresent();
};

/**
 * Add a module to load before Angular whenever Protractor.get is called.
 * Modules will be registered after existing modules already on the page,
 * so any module registered here will override preexisting modules with the same
 * name.
 *
 * @example
 * browser.addMockModule('modName', function() {
 *   angular.module('modName', []).value('foo', 'bar');
 * });
 *
 * @param {!string} name The name of the module to load or override.
 * @param {!string|Function} script The JavaScript to load the module.
 * @param {...*} varArgs Any additional arguments will be provided to
 *     the script and may be referenced using the `arguments` object.
 */
Protractor.prototype.addMockModule = function(name, script) {
  var moduleArgs = Array.prototype.slice.call(arguments, 2);

  this.mockModules_.push({
    name: name,
    script: script,
    args: moduleArgs
  });
};

/**
 * Clear the list of registered mock modules.
 */
Protractor.prototype.clearMockModules = function() {
  this.mockModules_ = [];
  this.addBaseMockModules_();
};

/**
 * Remove a registered mock module.
 *
 * @example
 * browser.removeMockModule('modName');
 *
 * @param {!string} name The name of the module to remove.
 */
Protractor.prototype.removeMockModule = function(name) {
  for (var i = 0; i < this.mockModules_.length; ++i) {
    if (this.mockModules_[i].name == name) {
      this.mockModules_.splice(i, 1);
    }
  }
};

/**
 * Get a list of the current mock modules.
 *
 * @return {Array.<!string|Function>}
 */
Protractor.prototype.getRegisteredMockModules = function() {
  return this.mockModules_.map(function(module) {
    return module.script;
  });
};

/**
 * Add the base mock modules used for all Protractor tests.
 *
 * @private
 */
Protractor.prototype.addBaseMockModules_ = function() {
  this.addMockModule('protractorBaseModule_', function() {
    angular.module('protractorBaseModule_', []).
        config(['$compileProvider', function($compileProvider) {
          if ($compileProvider.debugInfoEnabled) {
            $compileProvider.debugInfoEnabled(true);
          }
        }]);
  });
};

/**
 * @see webdriver.WebDriver.get
 *
 * Navigate to the given destination and loads mock modules before
 * Angular. Assumes that the page being loaded uses Angular.
 * If you need to access a page which does not have Angular on load, use
 * the wrapped webdriver directly.
 *
 * @example
 * browser.get('https://angularjs.org/');
 * expect(browser.getCurrentUrl()).toBe('https://angularjs.org/');
 *
 * @param {string} destination Destination URL.
 * @param {number=} opt_timeout Number of milliseconds to wait for Angular to
 *     start.
 */
Protractor.prototype.get = function(destination, opt_timeout) {
  var timeout = opt_timeout ? opt_timeout : this.getPageTimeout;
  var self = this;

  destination = this.baseUrl.indexOf('file://') === 0 ?
    this.baseUrl + destination : url.resolve(this.baseUrl, destination);
  var msg = function(str) {
    return 'Protractor.get(' + destination + ') - ' + str;
  };

  if (this.ignoreSynchronization) {
    return this.driver.get(destination);
  }

  this.driver.get(this.resetUrl);
  this.executeScript_(
      'window.name = "' + DEFER_LABEL + '" + window.name;' +
      'window.location.replace("' + destination + '");',
      msg('reset url'));

  // We need to make sure the new url has loaded before
  // we try to execute any asynchronous scripts.
  this.driver.wait(function() {
    return self.executeScript_('return window.location.href;', msg('get url')).
        then(function(url) {
          return url !== self.resetUrl;
        }, function(err) {
          if (err.code == 13) {
            // Ignore the error, and continue trying. This is because IE
            // driver sometimes (~1%) will throw an unknown error from this
            // execution. See https://github.com/angular/protractor/issues/841
            // This shouldn't mask errors because it will fail with the timeout
            // anyway.
            return false;
          } else {
            throw err;
          }
        });
  }, timeout,
  'waiting for page to load for ' + timeout + 'ms');

  // Make sure the page is an Angular page.
  self.executeAsyncScript_(clientSideScripts.testForAngular,
      msg('test for angular'),
      Math.floor(timeout / 1000)).
      then(function(angularTestResult) {
        var hasAngular = angularTestResult[0];
        if (!hasAngular) {
          var message = angularTestResult[1];
          throw new Error('Angular could not be found on the page ' +
              destination + ' : ' + message);
        }
      }, function(err) {
        throw 'Error while running testForAngular: ' + err.message;
      });

  // At this point, Angular will pause for us until angular.resumeBootstrap
  // is called.
  var moduleNames = [];
  for (var i = 0; i < this.mockModules_.length; ++i) {
    var mockModule = this.mockModules_[i];
    var name = mockModule.name;
    moduleNames.push(name);
    var executeScriptArgs = [mockModule.script, msg('add mock module ' + name)].
        concat(mockModule.args);
    this.executeScript_.apply(this, executeScriptArgs).
        then(null, function(err) {
          throw 'Error while running module script ' + name +
              ': ' + err.message;
        });
  }

  return this.executeScript_(
      'angular.resumeBootstrap(arguments[0]);',
      msg('resume bootstrap'),
      moduleNames);
};

/**
 * @see webdriver.WebDriver.refresh
 *
 * Makes a full reload of the current page and loads mock modules before
 * Angular. Assumes that the page being loaded uses Angular.
 * If you need to access a page which does not have Angular on load, use
 * the wrapped webdriver directly.
 *
 * @param {number=} opt_timeout Number of seconds to wait for Angular to start.
 */
Protractor.prototype.refresh = function(opt_timeout) {
  var timeout = opt_timeout || 10;
  var self = this;

  if (self.ignoreSynchronization) {
    return self.driver.navigate().refresh();
  }

  return self.executeScript_(
      'return window.location.href',
      'Protractor.refresh() - getUrl').then(function(href) {
        return self.get(href, timeout);
      });
};

/**
 * Mixin navigation methods back into the navigation object so that
 * they are invoked as before, i.e. driver.navigate().refresh()
 */
Protractor.prototype.navigate = function() {
  var nav = this.driver.navigate();
  mixin(nav, this, 'refresh');
  return nav;
};

/**
 * Browse to another page using in-page navigation.
 *
 * @example
 * browser.get('http://angular.github.io/protractor/#/tutorial');
 * browser.setLocation('api');
 * expect(browser.getCurrentUrl())
 *     .toBe('http://angular.github.io/protractor/#/api');
 *
 * @param {string} url In page URL using the same syntax as $location.url()
 * @return {!webdriver.promise.Promise} A promise that will resolve once
 *    page has been changed.
 */
Protractor.prototype.setLocation = function(url) {
  this.waitForAngular();
  return this.executeScript_(clientSideScripts.setLocation,
    'Protractor.setLocation()', this.rootEl, url).then(function(browserErr) {
      if (browserErr) {
        throw 'Error while navigating to \'' + url + '\' : ' +
            JSON.stringify(browserErr);
      }
    });
};

/**
 * Returns the current absolute url from AngularJS.
 *
 * @example
 * browser.get('http://angular.github.io/protractor/#/api');
 * expect(browser.getLocationAbsUrl())
 *     .toBe('http://angular.github.io/protractor/#/api');
 */
Protractor.prototype.getLocationAbsUrl = function() {
  this.waitForAngular();
  return this.executeScript_(clientSideScripts.getLocationAbsUrl,
      'Protractor.getLocationAbsUrl()', this.rootEl);
};

/**
 * Adds a task to the control flow to pause the test and inject helper functions
 * into the browser, so that debugging may be done in the browser console.
 *
 * This should be used under node in debug mode, i.e. with
 * protractor debug <configuration.js>
 *
 * @example
 * While in the debugger, commands can be scheduled through webdriver by
 * entering the repl:
 *   debug> repl
 *   Press Ctrl + C to leave rdebug repl
 *   > ptor.findElement(protractor.By.input('user').sendKeys('Laura'));
 *   > ptor.debugger();
 *   debug> c
 *
 * This will run the sendKeys command as the next task, then re-enter the
 * debugger.
 */
Protractor.prototype.debugger = function() {
  // jshint debug: true
  this.driver.executeScript(clientSideScripts.installInBrowser);
  webdriver.promise.controlFlow().execute(function() { debugger; },
                                          'add breakpoint to control flow');
};

/**
 * Helper function to:
 *  1) Set up helper functions for debugger clients to call on (e.g.
 *     getControlFlowText, execute code, get autocompletion).
 *  2) Enter process into debugger mode. (i.e. process._debugProcess).
 *  3) Invoke the debugger client specified by debuggerClientPath.
 *
 * @param {string=} debuggerClientPath Absolute path of debugger client to use
 * @param {number=} opt_debugPort Optional port to use for the debugging process
 */
Protractor.prototype.initDebugger_ = function(debuggerClientPath, opt_debugPort) {
  // Patch in a function to help us visualize what's going on in the control
  // flow.
  webdriver.promise.ControlFlow.prototype.getControlFlowText = function() {
    var descriptions = [];

    var getDescriptions = function(frameOrTask, descriptions) {
      if (frameOrTask.getDescription) {
        var stacktrace = frameOrTask.snapshot_.getStacktrace();
        stacktrace = stacktrace ? stacktrace.join('\n').trim() : '';
        descriptions.push({
          description: frameOrTask.getDescription(),
          stack: helper.filterStackTrace(stacktrace)
        });
      } else {
        for (var i = 0; i < frameOrTask.children_.length; ++i) {
          getDescriptions(frameOrTask.children_[i], descriptions);
        }
      }
    };
    if (this.history_.length) {
      getDescriptions(this.history_[this.history_.length - 1], descriptions);
    }
    if (this.activeFrame_.getPendingTask()) {
      getDescriptions(this.activeFrame_.getPendingTask(), descriptions);
    }
    getDescriptions(this.activeFrame_.getRoot(), descriptions);
    var asString = '-- WebDriver control flow schedule \n';
    for (var i = 0; i < descriptions.length; ++i) {
      asString += ' |- ' + descriptions[i].description;
      if (descriptions[i].stack) {
        asString += '\n |---' + descriptions[i].stack.replace(/\n/g, '\n |---');
      }
      if (i != (descriptions.length - 1)) {
        asString += '\n';
      }
    }
    return asString;
  };

  if (opt_debugPort) {
    process.debugPort = opt_debugPort;
  }

  // Call this private function instead of sending SIGUSR1 because Windows.
  process._debugProcess(process.pid);

  var flow = webdriver.promise.controlFlow();
  var pausePromise = flow.execute(function() {
    log.puts('Starting WebDriver debugger in a child process. Pause is ' +
        'still beta, please report issues at github.com/angular/protractor\n');
    var nodedebug = require('child_process').
        fork(debuggerClientPath, [process.debugPort]);
    process.on('exit', function() {
      nodedebug.kill('SIGTERM');
    });
  });

  var vm_ = require('vm');
  var browserUnderDebug = this;

  // Helper used only by debuggers at './debugger/modes/*.js' to insert code
  // into the control flow.
  // In order to achieve this, we maintain a promise at the top of the control
  // flow, so that we can insert frames into it.
  // To be able to simulate callback/asynchronous code, we poll this object
  // for an result at every run of DeferredExecutor.execute.
  this.dbgCodeExecutor_ = {
    execPromise_: pausePromise, // Promise pointing to current stage of flow.
    execPromiseResult_: undefined, // Return value of promise.
    execPromiseError_: undefined, // Error from promise.

    // A dummy repl server to make use of its completion function.
    replServer_: require('repl').start({
      input: {on: function() {}, resume: function() {}}, // dummy readable stream
      output: {write: function() {}} // dummy writable stream
    }),

    // Execute a function, which could yield a value or a promise,
    // and allow its result to be accessed synchronously
    execute_: function(execFn_) {
      var self = this;
      self.execPromiseResult_ = self.execPromiseError_ = undefined;

      self.execPromise_ = self.execPromise_.
          then(function() {
            var result = execFn_();
            if (webdriver.promise.isPromise(result)) {
              return result.then(function(val) {return val;});
            } else {
              return result;
            }
          }).then(function(result) {
            self.execPromiseResult_ = result;
          }, function(err) {
            self.execPromiseError_ = err;
          });

      // This dummy command is necessary so that the DeferredExecutor.execute
      // break point can find something to stop at instead of moving on to the
      // next real command.
      self.execPromise_.then(function() {
        return browserUnderDebug.executeScript_('', 'empty debugger hook');
      });
    },

    // Execute a piece of code.
    execute: function(code) {
      var execFn_ = function() {
        // Run code through vm so that we can maintain a local scope which is
        // isolated from the rest of the execution.
        return vm_.runInThisContext(code);
      };
      this.execute_(execFn_);
    },

    // Autocomplete for a line.
    complete: function(line) {
      var self = this;
      var execFn_ = function() {
        var deferred = webdriver.promise.defer();
        self.replServer_.complete(line, function(err, res) {
          deferred.fulfill(res, err);
        });
        return deferred;
      };
      this.execute_(execFn_);
    },

    // Code finished executing.
    resultReady: function() {
      return !this.execPromise_.isPending();
    },

    // Get asynchronous results synchronously.
    // This will throw if result is not ready.
    getResult: function() {
      if (!this.resultReady()) {
        throw new Error('Result not ready');
      }
      if (this.execPromiseError_) {
        throw this.execPromiseError_;
      }

      return JSON.stringify(this.execPromiseResult_);
    }
  };

  global.list = function(locator) {
    /* globals browser */
    return browser.findElements(locator).then(function(arr) {
      var found = [];
      for (var i = 0; i < arr.length; ++i) {
        arr[i].getText().then(function(text) {
          found.push(text);
        });
      }
      return found;
    });
  };

  flow.timeout(1000, 'waiting for debugger to attach');
};

/**
 * Beta (unstable) enterRepl function for entering the repl loop from
 * any point in the control flow. Use browser.enterRepl() in your test.
 * Does not require changes to the command line (no need to add 'debug').
 * Note, if you are wrapping your own instance of Protractor, you must
 * expose globals 'browser' and 'protractor' for pause to work.
 *
 * @example
 * element(by.id('foo')).click();
 * browser.enterRepl();
 * // Execution will stop before the next click action.
 * element(by.id('bar')).click();
 *
 * @param {number=} opt_debugPort Optional port to use for the debugging process
 */
Protractor.prototype.enterRepl = function(opt_debugPort) {
  var debuggerClientPath = __dirname + '/debugger/clients/explorer.js';
  this.initDebugger_(debuggerClientPath, opt_debugPort);
};

/**
 * Beta (unstable) pause function for debugging webdriver tests. Use
 * browser.pause() in your test to enter the protractor debugger from that
 * point in the control flow.
 * Does not require changes to the command line (no need to add 'debug').
 * Note, if you are wrapping your own instance of Protractor, you must
 * expose globals 'browser' and 'protractor' for pause to work.
 *
 * @example
 * element(by.id('foo')).click();
 * browser.pause();
 * // Execution will stop before the next click action.
 * element(by.id('bar')).click();
 *
 * @param {number=} opt_debugPort Optional port to use for the debugging process
 */
Protractor.prototype.pause = function(opt_debugPort) {
  var debuggerClientPath = __dirname + '/debugger/clients/wddebugger.js';
  this.initDebugger_(debuggerClientPath, opt_debugPort);
};

/**
 * Create a new instance of Protractor by wrapping a webdriver instance.
 *
 * @param {webdriver.WebDriver} webdriver The configured webdriver instance.
 * @param {string=} opt_baseUrl A URL to prepend to relative gets.
 * @return {Protractor}
 */
exports.wrapDriver = function(webdriver, opt_baseUrl, opt_rootElement) {
  return new Protractor(webdriver, opt_baseUrl, opt_rootElement);
};
