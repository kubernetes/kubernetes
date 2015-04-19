// Copyright 2013 Selenium committers
// Copyright 2013 Software Freedom Conservancy
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//     You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

'use strict';

var assert = require('assert');

var build = require('./build'),
    webdriver = require('../..'),
    flow = webdriver.promise.controlFlow(),
    _base = require('../../_base'),
    testing = require('../../testing'),
    fileserver = require('./fileserver'),
    seleniumserver = require('./seleniumserver');


var Browser = {
  ANDROID: 'android',
  CHROME: 'chrome',
  IE: 'internet explorer',
  // Shorthand for IPAD && IPHONE when using the browsers predciate.
  IOS: 'iOS',
  IPAD: 'iPad',
  IPHONE: 'iPhone',
  FIREFOX: 'firefox',
  OPERA: 'opera',
  PHANTOMJS: 'phantomjs',
  SAFARI: 'safari',

  // Browsers that should always be tested via the java Selenium server.
  REMOTE_CHROME: 'remote.chrome',
  REMOTE_FIREFOX: 'remote.firefox',
  REMOTE_PHANTOMJS: 'remote.phantomjs'
};


/**
 * Browsers with native support.
 * @type {!Array.<string>}
 */
var NATIVE_BROWSERS = [
  Browser.CHROME,
  Browser.FIREFOX,
  Browser.PHANTOMJS
];


var browsersToTest = (function() {
  var browsers = process.env['SELENIUM_BROWSERS'] || Browser.FIREFOX;
  browsers = browsers.split(',');
  browsers.forEach(function(browser) {
    if (browser === Browser.IOS) {
      throw Error('Invalid browser name: ' + browser);
    }

    for (var name in Browser) {
      if (Browser.hasOwnProperty(name) && Browser[name] === browser) {
        return;
      }
    }

    throw Error('Unrecognized browser: ' + browser);
  });
  return browsers;
})();


/**
 * Creates a predicate function that ignores tests for specific browsers.
 * @param {string} currentBrowser The name of the current browser.
 * @param {!Array.<!Browser>} browsersToIgnore The browsers to ignore.
 * @return {function(): boolean} The predicate function.
 */
function browsers(currentBrowser, browsersToIgnore) {
  return function() {
    var checkIos =
        currentBrowser === Browser.IPAD || currentBrowser === Browser.IPHONE;
    return browsersToIgnore.indexOf(currentBrowser) != -1 ||
        (checkIos && browsersToIgnore.indexOf(Browser.IOS) != -1);
  };
}


/**
 * @param {string} browserName The name to use.
 * @param {remote.DriverService} server The server to use, if any.
 * @constructor
 */
function TestEnvironment(browserName, server) {
  var name = browserName;
  if (name.lastIndexOf('remote.', 0) == 0) {
    name = name.substring('remote.'.length);
  }

  var autoCreate = true;
  this.__defineGetter__(
      'autoCreateDriver', function() { return autoCreate; });
  this.__defineSetter__(
      'autoCreateDriver', function(auto) { autoCreate = auto; });

  this.__defineGetter__('browser', function() { return name; });

  var driver;
  this.__defineGetter__('driver', function() { return driver; });
  this.__defineSetter__('driver', function(d) {
    if (driver) throw Error('Driver already created');
    driver = d;
  });

  this.browsers = function(var_args) {
    var browsersToIgnore = Array.prototype.slice.apply(arguments, [0]);
    var remoteVariants = [];
    browsersToIgnore.forEach(function(browser) {
      if (browser.lastIndexOf('remote.', 0) === 0) {
        remoteVariants.push(browser.substring('remote.'.length));
      }
    });
    browsersToIgnore = browsersToIgnore.concat(remoteVariants);
    return browsers(browserName, browsersToIgnore);
  };

  this.builder = function() {
    assert.ok(!driver, 'Can only have one driver at a time');
    var builder = new webdriver.Builder();
    var realBuild = builder.build;

    builder.build = function() {
      builder.getCapabilities().
          set(webdriver.Capability.BROWSER_NAME, name);

      if (server) {
        builder.usingServer(server.address());
      }
      return driver = realBuild.call(builder);
    };

    return builder;
  };

  this.createDriver = function() {
    if (!driver) {
      driver = this.builder().build();
    }
    return driver;
  };

  this.refreshDriver = function() {
    if (driver) {
      driver.quit();
      driver = null;
    }
    this.createDriver();
  };

  this.dispose = function() {
    if (driver) {
      var d = driver;
      driver = null;
      return d.quit();
    }
  };
}


var seleniumServer;
var inSuite = false;


/**
 * Expands a function to cover each of the target browsers.
 * @param {function(!TestEnvironment)} fn The top level suite
 *     function.
 * @param {{browsers: !Array.<string>}=} opt_options Suite specific options.
 */
function suite(fn, opt_options) {
  assert.ok(!inSuite, 'You may not nest suite calls');
  inSuite = true;

  var suiteOptions = opt_options || {};
  var browsers = suiteOptions.browsers;
  if (browsers) {
    // Filter out browser specific tests when that browser is not currently
    // selected for testing.
    browsers = browsers.filter(function(browser) {
      if (browsersToTest.indexOf(browser) != -1) {
        return true;
      }
      return browsersToTest.indexOf(
          browser.substring('remote.'.length)) != -1;
    });
  } else {
    browsers = browsersToTest;
  }

  try {
    browsers.forEach(function(browser) {

      testing.describe('[' + browser + ']', function() {
        var serverToUse = null;

        if (NATIVE_BROWSERS.indexOf(browser) == -1) {
          serverToUse = seleniumServer;
          if (!serverToUse) {
            serverToUse = seleniumServer = new seleniumserver.Server();
          }
          testing.before(function() {
            // Starting the server may require a build, so disable timeouts.
            this.timeout(0);
            return seleniumServer.start(60 * 1000);
          });
        }

        var env = new TestEnvironment(browser, serverToUse);

        testing.beforeEach(function() {
          if (env.autoCreateDriver) {
            return env.createDriver().getSession();  // Catch start-up failures.
          }
        });

        testing.after(function() {
          return env.dispose();
        });

        fn(env);
      });
    });
  } finally {
    inSuite = false;
  }
}


// GLOBAL TEST SETUP


testing.before(fileserver.start);
testing.after(fileserver.stop);

if (_base.isDevMode() && browsersToTest.indexOf(Browser.FIREFOX) != -1) {
  testing.before(function() {
    return build.of('//javascript/firefox-driver:webdriver').onlyOnce().go();
  });
}

// Server is only started if required for a specific config.
testing.after(function() {
  if (seleniumServer) {
    seleniumServer.stop();
  }
});


// PUBLIC API


exports.suite = suite;
exports.after = testing.after;
exports.afterEach = testing.afterEach;
exports.before = testing.before;
exports.beforeEach = testing.beforeEach;
exports.it = testing.it;
exports.ignore = testing.ignore;

exports.Browser = Browser;
exports.Pages = fileserver.Pages;
exports.whereIs = fileserver.whereIs;
