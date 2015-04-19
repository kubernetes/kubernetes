WebDriver Providers for Protractor
==================================

DriverProviders define ways that the Protractor runner can connect to
WebDriver.

Each file exports a function which takes in the configuration as a parameter and returns a new DriverProvider, which has the following interface:

```js
/**
 * @return {q.promise} A promise which will resolve when the environment is
 *     ready to test.
 */
DriverProvider.prototype.setupEnv

/**
 * @return {Array.<webdriver.WebDriver>} Array of existing webdriver instances.
 */
DriverProvider.prototype.getExistingDrivers

/**
 * @return {webdriver.WebDriver} A new setup driver instance.
 */
DriverProvider.prototype.getNewDriver

/**
 * @param {webdriver.WebDriver} The driver instance to quit.
 */
DriverProvider.prototype.quitDriver

/**
 * @return {q.promise} A promise which will resolve when the environment
 *     is down.
 */
DriverProvider.prototype.teardownEnv

/**
 * This is an optional function. If defined, it will be called with the final
 * status of the test suite (pass/fail) once it is completed.
 *
 * @param {{passed: boolean}}
 * @return {q.promise} A promise that will resolve when the update is complete.
 */
DriverProvider.prototype.updateJob
```

Requirements
------------

 - `setupEnv` will be called before the test framework is loaded, so any
 pre-work which might cause timeouts on the first test should be done there. 
 `getNewDriver` will be called once right after `setupEnv` to generate the
 initial driver, and possibly during the middle of the test if users request
 additional browsers.

 - `teardownEnv` should call the driver's `quit` method.
