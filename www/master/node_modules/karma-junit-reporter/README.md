# karma-junit-reporter

> Reporter for the JUnit XML format.

## Installation

The easiest way is to keep `karma-junit-reporter` as a devDependency in your `package.json`.
```json
{
  "devDependencies": {
    "karma": "~0.10",
    "karma-junit-reporter": "~0.1"
  }
}
```

You can simple do it by:
```bash
npm install karma-junit-reporter --save-dev
```

## Configuration
```js
// karma.conf.js
module.exports = function(config) {
  config.set({
    reporters: ['progress', 'junit'],

    // the default configuration
    junitReporter: {
      outputFile: 'test-results.xml',
      suite: ''
    }
  });
};
```

You can pass list of reporters as a CLI argument too:
```bash
karma start --reporters junit,dots
```

----

For more information on Karma see the [homepage].


[homepage]: http://karma-runner.github.com
