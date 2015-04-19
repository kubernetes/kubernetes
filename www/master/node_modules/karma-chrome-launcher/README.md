# karma-chrome-launcher

> Launcher for Google Chrome and Google Chrome Canary.

## Installation

The easiest way is to keep `karma-chrome-launcher` as a devDependency in your `package.json`.
```json
{
  "devDependencies": {
    "karma": "~0.10",
    "karma-chrome-launcher": "~0.1"
  }
}
```

You can simple do it by:
```bash
npm install karma-chrome-launcher --save-dev
```

## Configuration
```js
// karma.conf.js
module.exports = function(config) {
  config.set({
    browsers: ['Chrome', 'Chrome_without_security'],

    // you can define custom flags
    customLaunchers: {
      Chrome_without_security: {
        base: 'Chrome',
        flags: ['--disable-web-security']
      }
    }
  });
};
```

You can pass list of browsers as a CLI argument too:
```bash
karma start --browsers Chrome,Chrome_without_security
```

----

For more information on Karma see the [homepage].


[homepage]: http://karma-runner.github.com
