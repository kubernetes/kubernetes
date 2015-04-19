# timed-out [![Build Status](https://travis-ci.org/floatdrop/timed-out.svg?branch=master)](https://travis-ci.org/floatdrop/timed-out)

> Timeout HTTP/HTTPS requests

Emit Error object with `code` property equal `ETIMEDOUT` or `ESOCKETTIMEDOUT` when ClientRequest is hanged.

## Usage

```js
var get = require('http').get;
var timeout = require('timed-out');

var req = get('http://www.google.ru');
timeout(req, 2000); // Set 2 seconds limit
```

### API

#### timedout(request, time)

##### request

*Required*  
Type: [`ClientRequest`](http://nodejs.org/api/http.html#http_class_http_clientrequest)

The request to watch on.

##### time

*Required*  
Type: `number`

Time in milliseconds before errors will be emitted and `request.abort()` call happens.

## License

MIT © [Vsevolod Strukchinsky](floatdrop@gmail.com)
