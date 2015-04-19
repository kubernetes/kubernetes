# karma-html2js-preprocessor [![Build Status](https://travis-ci.org/karma-runner/karma-html2js-preprocessor.png?branch=master)](https://travis-ci.org/karma-runner/karma-html2js-preprocessor)

> Preprocessor for converting HTML files into JS strings.

*Note:* If you are using [AngularJS](http://angularjs.org/), check out [karma-ng-html2js-preprocessor](https://github.com/karma-runner/karma-ng-html2js-preprocessor).

## Installation

**This plugin ships with Karma by default, so you don't need to install it, it should just work ;-)**

The easiest way is to keep `karma-html2js-preprocessor` as a devDependency in your `package.json`.
```json
{
  "devDependencies": {
    "karma": "~0.10",
    "karma-html2js-preprocessor": "~0.1"
  }
}
```

You can simple do it by:
```bash
npm install karma-html2js-preprocessor --save-dev
```

## Configuration
Following code shows the default configuration...
```js
// karma.conf.js
module.exports = function(config) {
  config.set({
    preprocessors: {
      '**/*.html': ['html2js']
    },

    files: [
      '*.js',
      '*.html'
    ]
  });
};
```

## How does it work ?

This preprocessor converts HTML files into JS strings and publishes them in the global `window.__html__`, so that you can use these for testing DOM operations.

For instance this `template.html`...
```html
<div>something</div>
```
... will be served as `template.html.js`:
```js
window.__html__ = window.__html__ || {};
window.__html__['template.html'] = '<div>something</div>';
```

See the [end2end test](https://github.com/karma-runner/karma/tree/master/test/e2e/html2js) for a complete example.

----

For more information on Karma see the [homepage].


[homepage]: http://karma-runner.github.com
