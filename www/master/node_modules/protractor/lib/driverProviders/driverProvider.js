/**
 *  This is a base driver provider class.
 *  It is responsible for setting up the account object, tearing
 *  it down, and setting up the driver correctly.
 */

var webdriver = require('selenium-webdriver'),
    q = require('q');


var DriverProvider = function(config) {
  this.config_ = config;
  this.drivers_ = [];
};


/**
 * Get all existing drivers.
 *
 * @public
 * @return array of webdriver instances
 */
DriverProvider.prototype.getExistingDrivers = function() {
  return this.drivers_.slice(); // Create a shallow copy
};


/**
 * Create a new driver.
 *
 * @public
 * @return webdriver instance
 */
DriverProvider.prototype.getNewDriver = function() {
  var newDriver = new webdriver.Builder().
      usingServer(this.config_.seleniumAddress).
      withCapabilities(this.config_.capabilities).
      build();
  this.drivers_.push(newDriver);
  return newDriver;
};


/**
 * Quit a driver.
 *
 * @public
 * @param webdriver instance
 */
DriverProvider.prototype.quitDriver = function(driver) {
  var driverIndex = this.drivers_.indexOf(driver);
  if (driverIndex >= 0) {
    this.drivers_.splice(driverIndex, 1);
  }

  var deferred = q.defer();
  driver.getSession().then(function(session_) {
    if (session_) {
      driver.quit().then(function() {
        deferred.resolve();
      });
    } else {
      deferred.resolve();
    }
  });
  return deferred.promise;
};


/**
 * Teardown and destroy the environment and do any associated cleanup.
 * Shuts down the drivers.
 *
 * @public
 * @return {q.promise} A promise which will resolve when the environment
 *     is down.
 */
DriverProvider.prototype.teardownEnv = function() {
  return q.all(this.drivers_.map(this.quitDriver.bind(this)));
};


module.exports = DriverProvider;
