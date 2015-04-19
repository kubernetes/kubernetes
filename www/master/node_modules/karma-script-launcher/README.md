# karma-scripts-launcher

> Launcher for a shell script.

This plugin allows you to use a shell script as a browser launcher. The script has to accept
a single argument - the url that the browser should open.

## Installation

**This plugin ships with Karma by default, so you don't need to install it, it should just work ;-)**

The easiest way is to keep `karma-scripts-launcher` as a devDependency in your `package.json`.
```json
{
  "devDependencies": {
    "karma": "~0.10",
    "karma-scripts-launcher": "~0.1"
  }
}
```

You can simple do it by:
```bash
npm install karma-scripts-launcher --save-dev
```

## Configuration
```js
// karma.conf.js
module.exports = function(config) {
  config.set({
    browsers: ['/usr/local/bin/my-custom.sh'],
  });
};
```

You can pass list of browsers as a CLI argument too:
```bash
karma start --browsers /some/custom/script.sh
```

----

For more information on Karma see the [homepage].


[homepage]: http://karma-runner.github.com
