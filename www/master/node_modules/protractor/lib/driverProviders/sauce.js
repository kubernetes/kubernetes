/*
 * This is an implementation of the SauceLabs Driver Provider.
 * It is responsible for setting up the account object, tearing
 * it down, and setting up the driver correctly.
 */

var util = require('util'),
    log = require('../logger.js'),
    SauceLabs = require('saucelabs'),
    q = require('q'),
    DriverProvider = require('./driverProvider');


var SauceDriverProvider = function(config) {
  DriverProvider.call(this, config);
  this.sauceServer_ = {};
};
util.inherits(SauceDriverProvider, DriverProvider);


/**
 * Hook to update the sauce job.
 * @public
 * @param {Object} update
 * @return {q.promise} A promise that will resolve when the update is complete.
 */
SauceDriverProvider.prototype.updateJob = function(update) {

  var self = this;
  var deferredArray = this.drivers_.map(function(driver) {
    var deferred = q.defer();
    driver.getSession().then(function(session) {
      log.puts('SauceLabs results available at http://saucelabs.com/jobs/' +
          session.getId());
      self.sauceServer_.updateJob(session.getId(), update, function(err) {
        if (err) {
          throw new Error(
            'Error updating Sauce pass/fail status: ' + util.inspect(err)
          );
        }
        deferred.resolve();
      });
    });
    return deferred.promise;
  });
  return q.all(deferredArray);
};

/**
 * Configure and launch (if applicable) the object's environment.
 * @public
 * @return {q.promise} A promise which will resolve when the environment is
 *     ready to test.
 */
SauceDriverProvider.prototype.setupEnv = function() {
  var deferred = q.defer();
  this.sauceServer_ = new SauceLabs({
    username: this.config_.sauceUser,
    password: this.config_.sauceKey
  });
  this.config_.capabilities.username = this.config_.sauceUser;
  this.config_.capabilities.accessKey = this.config_.sauceKey;
  var auth = 'http://' + this.config_.sauceUser + ':' +
    this.config_.sauceKey + '@';
  this.config_.seleniumAddress = auth +
      (this.config_.sauceSeleniumAddress ? this.config_.sauceSeleniumAddress :
      'ondemand.saucelabs.com:80/wd/hub');

  // Append filename to capabilities.name so that it's easier to identify tests.
  if (this.config_.capabilities.name &&
      this.config_.capabilities.shardTestFiles) {
    this.config_.capabilities.name += (
        ':' + this.config_.specs.toString().replace(/^.*[\\\/]/, ''));
  }

  log.puts('Using SauceLabs selenium server at ' +
      this.config_.seleniumAddress.replace(/\/\/.+@/, '//'));
  deferred.resolve();
  return deferred.promise;
};

// new instance w/ each include
module.exports = function(config) {
  return new SauceDriverProvider(config);
};
