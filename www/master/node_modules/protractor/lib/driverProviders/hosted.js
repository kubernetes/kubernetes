/*
 *  This is an implementation of the Hosted Driver Provider.
 *  It is responsible for setting up the account object, tearing
 *  it down, and setting up the driver correctly.
 */

var util = require('util'),
    q = require('q'),
    DriverProvider = require('./driverProvider'),
    log = require('../logger');

var HostedDriverProvider = function(config) {
  DriverProvider.call(this, config);
};
util.inherits(HostedDriverProvider, DriverProvider);

/**
 * Configure and launch (if applicable) the object's environment.
 * @public
 * @return {q.promise} A promise which will resolve when the environment is
 *     ready to test.
 */
HostedDriverProvider.prototype.setupEnv = function() {
  log.puts('Using the selenium server at ' +
      this.config_.seleniumAddress);
  return q.fcall(function() {});
};

// new instance w/ each include
module.exports = function(config) {
  return new HostedDriverProvider(config);
};
