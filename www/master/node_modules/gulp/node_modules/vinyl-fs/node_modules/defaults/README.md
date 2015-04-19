# defaults

A simple one level options merge utility

## install

`npm install defaults`

## use

```javascript

var defaults = require('defaults');

var handle = function(options, fn) {
  options = defaults(options, {
    timeout: 100
  });

  setTimeout(function() {
    fn(options);
  }, options.timeout);
}

handle({ timeout: 1000 }, function() {
  // we're here 1000 ms later
});

handle({ timeout: 10000 }, function() {
  // we're here 10s later
});

```

## summary

this module exports a function that takes 2 arguments: `options` and `defaults`.  When called, it overrides all of `undefined` properties in `options` with the clones of properties defined in `defaults`

Sidecases: if called with a falsy `options` value, options will be initialized to a new object before being merged onto.

## license

[MIT](LICENSE)
