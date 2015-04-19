# 1.0.0 -> 2.0.0
The option `nodeFlags` was renamed to `v8flags` for accuracy. It can now be a callback taking method that yields an array of flags, **or** an array literal.

# 0.11 -> 0.12
For the environment passed into the `launch` callback, `configNameRegex` has been renamed to `configNameSearch`.  It now returns an array of valid config names instead of a regular expression.

# 0.10 -> 0.11
The method signature for `launch` was changed in this version of Liftoff.

You must now provide your own options parser and pass your desired params directly into `launch` as the first argument.  The second argument is now the invocation callback that starts your application.

To replicate the default functionality of 0.10, use the following:
```js
const Liftoff = require('liftoff');
const MyApp = new Liftoff({name:'myapp'});
const argv = require('minimist')(process.argv.slice(2));
const invoke = function (env) {
  console.log('my environment is:', env);
  console.log('my cli options are:', argv);
  console.log('my liftoff config is:', this);
};
MyApp.launch({
  cwd: argv.cwd,
  configPath: argv.myappfile,
  require: argv.require,
  completion: argv.completion
}, invoke);
```
