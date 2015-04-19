Protractor Plugins
=================

Plugins extend Protractor's base features by using hooks during test
execution to gather more data and potentially modify the test output.

The Protractor API and available plugins are *BETA* and may change
without a major version bump.

This folder contains default plugins for Protractor.

Using Plugins
-------------

Plugins are enabled via your config file.

```javascript
// protractor.conf.js
exports.config = {

  // ... the rest of your config

  plugins: [{
    // The only required field for each plugin is the path to that
    // plugin's entry script.
    path: 'path/to/plugin/index.js',

    // Plugins may use additional options specified here. See the
    // individual plugin docs for more information.
    option1: 'foo',
    option2: 'bar'
  }]
};
```

Protractor contains built in plugins in the 'plugins' folder. An example of
using the 'ngHint' plugin is shown below.

```javascript
  plugins: [{
    path: 'node_modules/protractor/plugins/ngHint',
  }]
```

Finally, if your plugin is a node module, you may use it with the `package`
option. For example, if you did `npm install example-protractor-plugin` your
config would look like:

```javascript
  plugins: [{
    package: 'example-protractor-plugin',
  }]
```

Writing Plugins
---------------

Plugins are designed to work with any test framework (Jasmine, Mocha, etc),
so they use generic hooks which Protractor provides. Plugins may change
the output of Protractor by returning a results object.

Plugins are node modules which export an object with the following API:

```js
/*
 * Sets up plugins before tests are run. This is called after the WebDriver
 * session has been started, but before the test framework has been set up.
 *
 * @param {Object} config The plugin configuration object. Note that
 *     this is not the entire Protractor config object, just the
 *     entry in the plugins array for this plugin.
 *
 * @return Object If an object is returned, it is merged with the Protractor
 *     result object. May return a promise.
 */
exports.setup = function(config) {};

/*
 * This is called after the tests have been run, but before the WebDriver
 * session has been terminated.
 *
 * @param {Object} config The plugin configuration object.
 *
 * @return Object If an object is returned, it is merged with the Protractor
 *     result object. May return a promise.
 */
exports.teardown = function(config) {};

/*
 * Called after the test results have been finalized and any jobs have been
 * updated (if applicable).
 *
 * @param {Object} config The plugin configuration object.
 *
 * @return Return values are ignored.
 */
exports.postResults = function(config) {};

/**
 * Called after each test block (in Jasmine, this means an `it` block)
 * completes.
 *
 * @param {Object} config The plugin configuration object.
 * @param {boolean} passed True if the test passed.
 *
 * @return Object If an object is returned, it is merged with the Protractor
 *     result object. May return a promise.
 */
exports.postTest = function(config, passed) {};

/**
 * Used when reporting results.
 * @type {string}
 */
exports.name = '';
```

The protractor results object follows the format specified in
the [Framework documentation](../lib/frameworks/README.md).
