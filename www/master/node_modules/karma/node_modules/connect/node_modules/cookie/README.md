# cookie [![Build Status](https://secure.travis-ci.org/shtylman/node-cookie.png?branch=master)](http://travis-ci.org/shtylman/node-cookie) #

cookie is a basic cookie parser and serializer. It doesn't make assumptions about how you are going to deal with your cookies. It basically just provides a way to read and write the HTTP cookie headers.

See [RFC6265](http://tools.ietf.org/html/rfc6265) for details about the http header for cookies.

## how?

```
npm install cookie
```

```javascript
var cookie = require('cookie');

var hdr = cookie.serialize('foo', 'bar');
// hdr = 'foo=bar';

var cookies = cookie.parse('foo=bar; cat=meow; dog=ruff');
// cookies = { foo: 'bar', cat: 'meow', dog: 'ruff' };
```

## more

The serialize function takes a third parameter, an object, to set cookie options. See the RFC for valid values.

### path
> cookie path

### expires
> absolute expiration date for the cookie (Date object)

### maxAge
> relative max age of the cookie from when the client receives it (seconds)

### domain
> domain for the cookie

### secure
> true or false

### httpOnly
> true or false

