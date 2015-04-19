Framework Adapters for Protractor
=================================

Protractor can work with any test framework that is adapted here.

Each file details the adapter for one test framework. Each file must export a `run` function with the interface:

```js
/**
 * @param {Runner} runner The Protractor runner instance.
 * @param {Array.<string>} specs A list of absolute filenames.
 * @return {q.Promise} Promise resolved with the test results
 */
exports.run = function(runner, specs)
```

Requirements
------------

 - `runner.emit` must be called with `testPass` and `testFail` messages.

 - `runner.runTestPreparer` must be called before any tests are run.

 - `runner.getConfig().onComplete` must be called when tests are finished.

 - The returned promise must be resolved when tests are finished and it should return a results object. This object must have a `failedCount` property and optionally a `specResults` 
 object of the following structure:
 ```
   specResults = [{
     description: string,
     assertions: [{
       passed: boolean,
       errorMsg: string,
       stackTrace: string 
     }],
     duration: integer
   }]
 ```

Custom Frameworks
-----------------

If you have created/adapted a custom framework and want it added to 
Protractor core please send a PR so it can evaluated for addition as an 
official supported framework. In the meantime you can instruct Protractor
to use your own framework via the config file:

```js
exports.config = {
  // set to "custom" instead of jasmine/jasmine2/mocha/cucumber.
  framework: 'custom',
  // path relative to the current config file
  frameworkPath: './frameworks/my_custom_jasmine.js',
};
```

More on this at [referenceConf](../../docs/referenceConf.js) "The test framework" section.

**Disclaimer**: current framework interface can change without a major version bump.
