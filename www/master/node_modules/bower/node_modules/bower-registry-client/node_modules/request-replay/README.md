# request-replay [![Build Status](https://secure.travis-ci.org/IndigoUnited/node-request-replay.png)](http://travis-ci.org/IndigoUnited/node-request-replay.png)

Replays a [request](https://github.com/mikeal/request) when a network error occurs using the [retry](https://github.com/felixge/node-retry) module.

**DO NOT** use this module if you are piping `request` instances.
If you are listening to `data` events to buffer, beware that you must reset everything when a `replay` occurs.
This is why `pipping` is not supported.


## Installation

`$ npm install request-replay`


## Usage

```js
var fs = require('fs');
var request = require('request');
var replay = require('request-replay');

// Note that the options argument is optional
// Accepts the same options the retry module does and an additional
// errorCodes array that default to ['EADDRINFO', 'ETIMEDOUT', 'ECONNRESET', 'ESOCKETTIMEDOUT']
replay(request('http://google.com/doodle.png', function (err, response, body) {
    // Do things
}), {
    retries: 10,
    factor: 3
})
.on('replay', function (replay) {
    // "replay" is an object that contains some useful information
    console.log('request failed: ' + replay.error.code + ' ' + replay.error.message);
    console.log('replay nr: #' + replay.number);
    console.log('will retry in: ' + replay.delay + 'ms')
})
```

Note that the default retry options are modified to be more appropriate for requests:

* `retries`: The maximum amount of times to retry the operation. Default is `5`.
* `factor`: The exponential factor to use. Default is `2`.
* `minTimeout`: The amount of time before starting the first retry. Default is `2000`.
* `maxTimeout`: The maximum amount of time between two retries. Default is `35000`.
* `randomize`: Randomizes the timeouts by multiplying with a factor between `1` to `2`. Default is `true`.


## License

Released under the [MIT License](http://www.opensource.org/licenses/mit-license.php).
