/*
 * This is an mock implementation of the Driver Provider.
 * It returns a fake webdriver and never actually contacts a selenium
 * server.
 */
var webdriver = require('selenium-webdriver'),
    util = require('util'),
    q = require('q'),
    DriverProvider = require('./driverProvider');
/**
 * @constructor
 */
var MockExecutor = function() {
};

/**
 * @param {!webdriver.Command} command The command to execute.
 * @param {function(Error, !bot.response.ResponseObject=)} callback the function
 *     to invoke when the command response is ready.
 */
MockExecutor.prototype.execute = function(command, callback) {
  callback(null, {
    status: '0',
    value: 'test_response'
  });
};

var MockDriverProvider = function(config) {
  DriverProvider.call(this, config);
};
util.inherits(MockDriverProvider, DriverProvider);


/**
 * Configure and launch (if applicable) the object's environment.
 * @public
 * @return {q.promise} A promise which will resolve immediately.
 */
MockDriverProvider.prototype.setupEnv = function() {
  return q.fcall(function() {});
};


/**
 * Create a new driver.
 *
 * @public
 * @override
 * @return webdriver instance
 */
MockDriverProvider.prototype.getNewDriver = function() {
  var mockSession = new webdriver.Session('test_session_id', {});
  var newDriver = new webdriver.WebDriver(mockSession, new MockExecutor());
  this.drivers_.push(newDriver);
  return newDriver;
};

// new instance w/ each include
module.exports = function(config) {
  return new MockDriverProvider(config);
};
