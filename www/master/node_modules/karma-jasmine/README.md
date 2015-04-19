# karma-jasmine [![Build Status](https://travis-ci.org/karma-runner/karma-jasmine.png?branch=master)](https://travis-ci.org/karma-runner/karma-jasmine)

> Adapter for the [Jasmine](http://pivotal.github.io/jasmine/) testing framework.

## Installation

**This plugin ships with Karma by default, so you don't need to install it, it should just work ;-)**

The easiest way is to keep `karma-jasmine` as a devDependency in your `package.json`.
```json
{
  "devDependencies": {
    "karma": "~0.10",
    "karma-jasmine": "~0.1"
  }
}
```

You can simple do it by:
```bash
npm install karma-jasmine --save-dev
```

## Configuration
```js
// karma.conf.js
module.exports = function(config) {
  config.set({
    frameworks: ['jasmine'],

    files: [
      '*.js'
    ]
  });
};
```

----

For more information on Karma see the [homepage].


[homepage]: http://karma-runner.github.com
