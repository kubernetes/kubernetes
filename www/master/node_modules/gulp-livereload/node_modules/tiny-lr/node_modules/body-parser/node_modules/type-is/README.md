# type-is

[![NPM Version][npm-image]][npm-url]
[![NPM Downloads][downloads-image]][downloads-url]
[![Node.js Version][node-version-image]][node-version-url]
[![Build Status][travis-image]][travis-url]
[![Test Coverage][coveralls-image]][coveralls-url]

Infer the content-type of a request.

### Install

```sh
$ npm install type-is
```

## API

```js
var http = require('http')
var is   = require('type-is')

http.createServer(function (req, res) {
  var istext = is(req, ['text/*'])
  res.end('you ' + (istext ? 'sent' : 'did not send') + ' me text')
})
```

### type = is(request, types)

`request` is the node HTTP request. `types` is an array of types.

```js
// req.headers.content-type = 'application/json'

is(req, ['json'])             // 'json'
is(req, ['html', 'json'])     // 'json'
is(req, ['application/*'])    // 'application/json'
is(req, ['application/json']) // 'application/json'

is(req, ['html']) // false
```

### type = is.is(mediaType, types)

`mediaType` is the [media type](https://tools.ietf.org/html/rfc6838) string. `types` is an array of types.

```js
var mediaType = 'application/json'

is.is(mediaType, ['json'])             // 'json'
is.is(mediaType, ['html', 'json'])     // 'json'
is.is(mediaType, ['application/*'])    // 'application/json'
is.is(mediaType, ['application/json']) // 'application/json'

is.is(mediaType, ['html']) // false
```

### Each type can be:

- An extension name such as `json`. This name will be returned if matched.
- A mime type such as `application/json`.
- A mime type with a wildcard such as `*/json` or `application/*`. The full mime type will be returned if matched
- A suffix such as `+json`. This can be combined with a wildcard such as `*/vnd+json` or `application/*+json`. The full mime type will be returned if matched.

`false` will be returned if no type matches.

`null` will be returned if the request does not have a body.

## Examples

#### Example body parser

```js
var is = require('type-is');

function bodyParser(req, res, next) {
  if (!is.hasBody(req)) {
    return next()
  }

  switch (is(req, ['urlencoded', 'json', 'multipart'])) {
    case 'urlencoded':
      // parse urlencoded body
      throw new Error('implement urlencoded body parsing')
      break
    case 'json':
      // parse json body
      throw new Error('implement json body parsing')
      break
    case 'multipart':
      // parse multipart body
      throw new Error('implement multipart body parsing')
      break
    default:
      // 415 error code
      res.statusCode = 415
      res.end()
      return
  }
}
```

## License

[MIT](LICENSE)

[npm-image]: https://img.shields.io/npm/v/type-is.svg?style=flat
[npm-url]: https://npmjs.org/package/type-is
[node-version-image]: https://img.shields.io/node/v/type-is.svg?style=flat
[node-version-url]: http://nodejs.org/download/
[travis-image]: https://img.shields.io/travis/jshttp/type-is.svg?style=flat
[travis-url]: https://travis-ci.org/jshttp/type-is
[coveralls-image]: https://img.shields.io/coveralls/jshttp/type-is.svg?style=flat
[coveralls-url]: https://coveralls.io/r/jshttp/type-is?branch=master
[downloads-image]: https://img.shields.io/npm/dm/type-is.svg?style=flat
[downloads-url]: https://npmjs.org/package/type-is
