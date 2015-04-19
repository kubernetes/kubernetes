var protractor = require('./protractor'),
    webdriver = require('selenium-webdriver'),
    util = require('util'),
    q = require('q'),
    EventEmitter = require('events').EventEmitter,
    helper = require('./util'),
    log = require('./logger'),
    Plugins = require('./plugins');

/*
 * Runner is responsible for starting the execution of a test run and triggering
 * setup, teardown, managing config, etc through its various dependencies.
 *
 * The Protractor Runner is a node EventEmitter with the following events:
 * - testPass
 * - testFail
 * - testsDone
 *
 * @param {Object} config
 * @constructor
 */
var Runner = function(config) {
  log.set(config);
  this.preparer_ = null;
  this.driverprovider_ = null;
  this.config_ = config;

  if (config.v8Debug) {
    // Call this private function instead of sending SIGUSR1 because Windows.
    process._debugProcess(process.pid);
  }

  if (config.nodeDebug) {
    process._debugProcess(process.pid);
    var flow = webdriver.promise.controlFlow();

    flow.execute(function() {
      var nodedebug = require('child_process').fork('debug', ['localhost:5858']);
      process.on('exit', function() {
        nodedebug.kill('SIGTERM');
      });
      nodedebug.on('exit', function() {
        process.exit('1');
      });
    }, 'start the node debugger');
    flow.timeout(1000, 'waiting for debugger to attach');
  }

  if (config.capabilities && config.capabilities.seleniumAddress) {
    config.seleniumAddress = config.capabilities.seleniumAddress;
  }
  this.loadDriverProvider_(config);
  this.setTestPreparer(config.onPrepare);
};

util.inherits(Runner, EventEmitter);


/**
 * Registrar for testPreparers - executed right before tests run.
 * @public
 * @param {string/Fn} filenameOrFn
 */
Runner.prototype.setTestPreparer = function(filenameOrFn) {
  this.preparer_ = filenameOrFn;
};


/**
 * Executor of testPreparer
 * @public
 * @return {q.Promise} A promise that will resolve when the test preparers
 *     are finished.
 */
Runner.prototype.runTestPreparer = function() {
  return helper.runFilenameOrFn_(this.config_.configDir, this.preparer_);
};


/**
 * Grab driver provider based on type
 * @private
 *
 * Priority
 * 1) if directConnect is true, use that
 * 2) if seleniumAddress is given, use that
 * 3) if a Sauce Labs account is given, use that
 * 4) if a seleniumServerJar is specified, use that
 * 5) try to find the seleniumServerJar in protractor/selenium
 */
Runner.prototype.loadDriverProvider_ = function() {
  var runnerPath;
  if (this.config_.directConnect) {
    runnerPath = './driverProviders/direct';
  } else if (this.config_.seleniumAddress) {
    runnerPath = './driverProviders/hosted';
  } else if (this.config_.sauceUser && this.config_.sauceKey) {
    runnerPath = './driverProviders/sauce';
  } else if (this.config_.seleniumServerJar) {
    runnerPath = './driverProviders/local';
  } else if (this.config_.mockSelenium) {
    runnerPath = './driverProviders/mock';
  } else {
    runnerPath = './driverProviders/local';
  }

  this.driverprovider_ = require(runnerPath)(this.config_);
};


/**
 * Responsible for cleaning up test run and exiting the process.
 * @private
 * @param {int} Standard unix exit code
 */
Runner.prototype.exit_ = function(exitCode) {
  return helper.runFilenameOrFn_(
      this.config_.configDir, this.config_.onCleanUp, [exitCode]).
        then(function(returned) {
          if (typeof returned === 'number') {
            return returned;
          } else {
            return exitCode;
          }
        });
};


/**
 * Getter for the Runner config object
 * @public
 * @return {Object} config
 */
Runner.prototype.getConfig = function() {
  return this.config_;
};


/**
 * Get the control flow used by this runner.
 * @return {Object} WebDriver control flow.
 */
Runner.prototype.controlFlow = function() {
  return webdriver.promise.controlFlow();
};


/**
 * Sets up convenience globals for test specs
 * @private
 */
Runner.prototype.setupGlobals_ = function(browser_) {
  // Export protractor to the global namespace to be used in tests.
  global.protractor = protractor;
  global.browser = browser_;
  global.$ = browser_.$;
  global.$$ = browser_.$$;
  global.element = browser_.element;
  global.by = global.By = protractor.By;

  // Enable sourcemap support for stack traces.
  require('source-map-support').install();
  // Required by dart2js machinery.
  // https://code.google.com/p/dart/source/browse/branches/bleeding_edge/dart/sdk/lib/js/dart2js/js_dart2js.dart?spec=svn32943&r=32943#487
  global.DartObject = function(o) { this.o = o; };
};


/**
 * Create a new driver from a driverProvider. Then set up a
 * new protractor instance using this driver.
 * This is used to set up the initial protractor instances and any
 * future ones.
 *
 * @return {Protractor} a protractor instance.
 * @public
 */
Runner.prototype.createBrowser = function() {
  var config = this.config_;
  var driver = this.driverprovider_.getNewDriver();
  driver.manage().timeouts().setScriptTimeout(config.allScriptsTimeout);
  var browser_ = protractor.wrapDriver(driver,
      config.baseUrl, config.rootElement);
  browser_.params = config.params;
  if (config.getPageTimeout) {
    browser_.getPageTimeout = config.getPageTimeout;
  }
  var self = this;

  /**
   * Get the processed configuration object that is currently being run. This
   * will contain the specs and capabilities properties of the current runner
   * instance.
   *
   * @return {q.Promise} A promise which resolves to the capabilities object.
   */
  browser_.getProcessedConfig = function() {
    return webdriver.promise.fulfilled(config);
  };

  /**
   * Fork another instance of protractor for use in interactive tests.
   *
   * @param {boolean} opt_useSameUrl Whether to navigate to current url on creation
   * @param {boolean} opt_copyMockModules Whether to apply same mock modules on creation
   * @return {Protractor} a protractor instance.
   */
  browser_.forkNewDriverInstance = function(opt_useSameUrl, opt_copyMockModules) {
    var newBrowser = self.createBrowser();
    if (opt_copyMockModules) {
      newBrowser.mockModules_ = browser_.mockModules_;
    }
    if (opt_useSameUrl) {
      browser_.driver.getCurrentUrl().then(function(url) {
        newBrowser.get(url);
      });
    }
    return newBrowser;
  };
  return browser_;
};


/**
 * Final cleanup on exiting the runner.
 *
 * @return {q.Promise} A promise which resolves on finish.
 * @private
 */
Runner.prototype.shutdown_ = function() {
  return q.all(
      this.driverprovider_.getExistingDrivers().
          map(this.driverprovider_.quitDriver.bind(this.driverprovider_)));
};

/**
 * The primary workhorse interface. Kicks off the test running process.
 *
 * @return {q.Promise} A promise which resolves to the exit code of the tests.
 * @public
 */
Runner.prototype.run = function() {
  var self = this,
      testPassed,
      plugins,
      browser_;

  if (this.config_.framework !== 'explorer' && !this.config_.specs.length) {
    throw new Error('Spec patterns did not match any files.');
  }

  // 1) Setup environment
  //noinspection JSValidateTypes
  return this.driverprovider_.setupEnv().then(function() {
  // 2) Create a browser and setup globals
  }).then(function() {
    browser_ = self.createBrowser();
    self.setupGlobals_(browser_);
    return browser_.getSession().then(function(session) {
      log.debug('WebDriver session successfully started with capabilities ' +
          util.inspect(session.getCapabilities()));
    }, function(err) {
      log.error('Unable to start a WebDriver session.');
      throw err;
    });
  // 3) Setup plugins
  }).then(function() {
    plugins = new Plugins(self.config_);
    return plugins.setup();
  // 4) Execute test cases
  }).then(function(pluginSetupResults) {
    // Do the framework setup here so that jasmine and mocha globals are
    // available to the onPrepare function.
    var frameworkPath = '';
    if (self.config_.framework === 'jasmine') {
      frameworkPath = './frameworks/jasmine.js';
    } else if (self.config_.framework === 'jasmine2') {
      frameworkPath = './frameworks/jasmine2.js';
    } else if (self.config_.framework === 'mocha') {
      frameworkPath = './frameworks/mocha.js';
    } else if (self.config_.framework === 'cucumber') {
      frameworkPath = './frameworks/cucumber.js';
    } else if (self.config_.framework === 'debugprint') {
      // Private framework. Do not use.
      frameworkPath = './frameworks/debugprint.js';
    } else if (self.config_.framework === 'explorer') {
      // Private framework. Do not use.
      frameworkPath = './frameworks/explorer.js';
    } else if (self.config_.framework === 'custom') {
      if (!self.config_.frameworkPath) {
        throw new Error('When config.framework is custom, ' +
          'config.frameworkPath is required.');
      }
      frameworkPath = self.config_.frameworkPath;
    } else {
      throw new Error('config.framework (' + self.config_.framework +
          ') is not a valid framework.');
    }

    if (self.config_.restartBrowserBetweenTests) {
      var restartDriver = function() {
        // Note: because tests are not paused at this point, any async
        // calls here are not guaranteed to complete before the tests resume.
        self.driverprovider_.quitDriver(browser_.driver);
        // Copy mock modules, but do not navigate to previous URL.
        browser_ = browser_.forkNewDriverInstance(false, true);
        self.setupGlobals_(browser_);
      };

      self.on('testPass', restartDriver);
      self.on('testFail', restartDriver);
    }

    // We need to save these promises to make sure they're run, but we don't
    // want to delay starting the next test (because we can't, it's just
    // an event emitter).
    var pluginPostTestPromises = [];

    self.on('testPass', function() {
      pluginPostTestPromises.push(plugins.postTest(true));
    });
    self.on('testFail', function() {
      pluginPostTestPromises.push(plugins.postTest(false));
    });

    return require(frameworkPath).run(self, self.config_.specs).
        then(function(testResults) {
          return q.all(pluginPostTestPromises).then(function(postTestResultList) {
            var results = helper.joinTestLogs(pluginSetupResults, testResults);
            postTestResultList.forEach(function(postTestResult) {
              results = helper.joinTestLogs(results, postTestResult);
            });
            return results;
          });
        });
  // 5) Teardown plugins
  }).then(function(testResults) {
    var deferred = q.defer();
    plugins.teardown().then(function(pluginTeardownResults) {
      deferred.resolve(helper.joinTestLogs(testResults, pluginTeardownResults));
    }, function(error) {
      deferred.reject(error);
    });
    return deferred.promise;
  // 6) Teardown
  }).then(function(result) {
    self.emit('testsDone', result);
    testPassed = result.failedCount === 0;
    if (self.driverprovider_.updateJob) {
      return self.driverprovider_.updateJob({
            'passed': testPassed
          }).then(function() {
            return self.driverprovider_.teardownEnv();
          });
    } else {
      return self.driverprovider_.teardownEnv();
    }
  // 7) Let plugins do final cleanup
  }).then(function() {
    return plugins.postResults();
  // 8) Exit process
  }).then(function() {
    var exitCode = testPassed ? 0 : 1;
    return self.exit_(exitCode);
  }).fin(function() {
   return self.shutdown_();
  });
};

/**
 * Creates and returns a Runner instance
 *
 * @public
 * @param {Object} config
 */
module.exports = Runner;
