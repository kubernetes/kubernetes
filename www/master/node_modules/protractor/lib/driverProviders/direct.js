/*
 *  This is an implementation of the Direct Driver Provider.
 *  It is responsible for setting up the account object, tearing
 *  it down, and setting up the driver correctly.
 */

var webdriver = require('selenium-webdriver'),
    chrome = require('selenium-webdriver/chrome'),
    firefox = require('selenium-webdriver/firefox'),
    q = require('q'),
    fs = require('fs'),
    path = require('path'),
    util = require('util'),
    DriverProvider = require('./driverProvider'),
    log = require('../logger');

var DirectDriverProvider = function(config) {
  DriverProvider.call(this, config);
};
util.inherits(DirectDriverProvider, DriverProvider);

/**
 * Configure and launch (if applicable) the object's environment.
 * @public
 * @return {q.promise} A promise which will resolve when the environment is
 *     ready to test.
 */
DirectDriverProvider.prototype.setupEnv = function() {
  switch (this.config_.capabilities.browserName) {
    case 'chrome':
      log.puts('Using ChromeDriver directly...');
      break;
    case 'firefox':
      log.puts('Using FirefoxDriver directly...');
      break;
    default:
      throw new Error('browserName (' + this.config_.capabilities.browserName +
          ') is not supported with directConnect.');
  }
  return q.fcall(function() {});
};

/**
 * Create a new driver.
 *
 * @public
 * @override
 * @return webdriver instance
 */
DirectDriverProvider.prototype.getNewDriver = function() {
  var driver;
  switch (this.config_.capabilities.browserName) {
    case 'chrome':
      var chromeDriverFile = this.config_.chromeDriver ||
          path.resolve(__dirname, '../../selenium/chromedriver');

      // Check if file exists, if not try .exe or fail accordingly
      if (!fs.existsSync(chromeDriverFile)) {
        chromeDriverFile += '.exe';
        // Throw error if the client specified conf chromedriver and its not found
        if (!fs.existsSync(chromeDriverFile)) {
          throw new Error('Could not find chromedriver at ' +
            chromeDriverFile);
        }
      }

      var service = new chrome.ServiceBuilder(chromeDriverFile).build();
      driver = chrome.createDriver(
          new webdriver.Capabilities(this.config_.capabilities), service);
      break;
    case 'firefox':
      if (this.config_.firefoxPath) {
        this.config_.capabilities.firefox_binary = this.config_.firefoxPath;
      }
      driver = new firefox.Driver(this.config_.capabilities);
      break;
    default:
      throw new Error('browserName ' + this.config_.capabilities.browserName +
          'is not supported with directConnect.');
  }
  this.drivers_.push(driver);
  return driver;
};

// new instance w/ each include
module.exports = function(config) {
  return new DirectDriverProvider(config);
};
