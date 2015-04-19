# retry

Abstraction for exponential and custom retry strategies for failed operations.

## Installation

    npm install retry

## Current Status

This module has been tested and is ready to be used.

## Tutorial

The example below will retry a potentially failing `dns.resolve` operation
`10` times using an exponential backoff strategy. With the default settings, this
means the last attempt is made after `17 minutes and 3 seconds`.

``` javascript
var dns = require('dns');
var retry = require('retry');

function faultTolerantResolve(address, cb) {
  var operation = retry.operation();

  operation.attempt(function(currentAttempt) {
    dns.resolve(address, function(err, addresses) {
      if (operation.retry(err)) {
        return;
      }

      cb(err ? operation.mainError() : null, addresses);
    });
  });
}

faultTolerantResolve('nodejs.org', function(err, addresses) {
  console.log(err, addresses);
});
```

Of course you can also configure the factors that go into the exponential
backoff. See the API documentation below for all available settings.
currentAttempt is an int representing the number of attempts so far.

``` javascript
var operation = retry.operation({
  retries: 5,
  factor: 3,
  minTimeout: 1 * 1000,
  maxTimeout: 60 * 1000,
  randomize: true,
});
```

## API

### retry.operation([options])

Creates a new `RetryOperation` object. See the `retry.timeouts()` function
below for available `options`.

### retry.timeouts([options])

Returns an array of timeouts. All time `options` and return values are in
milliseconds. If `options` is an array, a copy of that array is returned.

`options` is a JS object that can contain any of the following keys:

* `retries`: The maximum amount of times to retry the operation. Default is `10`.
* `factor`: The exponential factor to use. Default is `2`.
* `minTimeout`: The number of milliseconds before starting the first retry. Default is `1000`.
* `maxTimeout`: The maximum number of milliseconds between two retries. Default is `Infinity`.
* `randomize`: Randomizes the timeouts by multiplying with a factor between `1` to `2`. Default is `false`.

The formula used to calculate the individual timeouts is:

```
var Math.min(random * minTimeout * Math.pow(factor, attempt), maxTimeout);
```

Have a look at [this article][article] for a better explanation of approach.

If you want to tune your `factor` / `times` settings to attempt the last retry
after a certain amount of time, you can use wolfram alpha. For example in order
to tune for `10` attempts in `5 minutes`, you can use this equation:

![screenshot](https://github.com/tim-kos/node-retry/raw/master/equation.gif)

Explaining the various values from left to right:

* `k = 0 ... 9`:  The `retries` value (10)
* `1000`: The `minTimeout` value in ms (1000)
* `x^k`: No need to change this, `x` will be your resulting factor
* `5 * 60 * 1000`: The desired total amount of time for retrying in ms (5 minutes)

To make this a little easier for you, use wolfram alpha to do the calculations:

[http://www.wolframalpha.com/input/?i=Sum%5B1000*x^k%2C+{k%2C+0%2C+9}%5D+%3D+5+*+60+*+1000]()

[article]: http://dthain.blogspot.com/2009/02/exponential-backoff-in-distributed.html

### new RetryOperation(timeouts)

Creates a new `RetryOperation` where `timeouts` is an array where each value is
a timeout given in milliseconds.

#### retryOperation.errors()

Returns an array of all errors that have been passed to
`retryOperation.retry()` so far.

#### retryOperation.mainError()

A reference to the error object that occured most frequently. Errors are
compared using the `error.message` property.

If multiple error messages occured the same amount of time, the last error
object with that message is returned.

If no errors occured so far, the value is `null`.

#### retryOperation.attempt(fn, timeoutOps)

Defines the function `fn` that is to be retried and executes it for the first
time right away. The `fn` function can receive an optional `currentAttempt` callback that represents the number of attempts to execute `fn` so far.

Optionally defines `timeoutOps` which is an object having a property `timeout` in miliseconds and a property `cb` callback function.
Whenever your retry operation takes longer than `timeout` to execute, the timeout callback function `cb` is called.


#### retryOperation.try(fn)

This is an alias for `retryOperation.attempt(fn)`. This is deprecated.

#### retryOperation.start(fn)

This is an alias for `retryOperation.attempt(fn)`. This is deprecated.

#### retryOperation.retry(error)

Returns `false` when no `error` value is given, or the maximum amount of retries
has been reached.

Otherwise it returns `true`, and retries the operation after the timeout for
the current attempt number.

#### retryOperation.attempts()

Returns an int representing the number of attempts it took to call `fn` before it was successful.

## License

retry is licensed under the MIT license.


#Changelog

0.6.0 Introduced optional timeOps parameter for the attempt() function which is an object having a property timeout in miliseconds and a property cb callback function. Whenever your retry operation takes longer than timeout to execute, the timeout callback function cb is called.

0.5.0 Some minor refactorings.

0.4.0 Changed retryOperation.try() to retryOperation.attempt(). Deprecated the aliases start() and try() for it.

0.3.0 Added retryOperation.start() which is an alias for retryOperation.try().

0.2.0 Added attempts() function and parameter to retryOperation.try() representing the number of attempts it took to call fn().
