# body-parser

[![NPM Version][npm-image]][npm-url]
[![NPM Downloads][downloads-image]][downloads-url]
[![Build Status][travis-image]][travis-url]
[![Test Coverage][coveralls-image]][coveralls-url]
[![Gratipay][gratipay-image]][gratipay-url]

Node.js body parsing middleware.

This does not handle multipart bodies, due to their complex and typically large nature. For multipart bodies, you may be interested in the following modules:

- [busboy](https://www.npmjs.org/package/busboy#readme) and [connect-busboy](https://www.npmjs.org/package/connect-busboy#readme)
- [multiparty](https://www.npmjs.org/package/multiparty#readme) and [connect-multiparty](https://www.npmjs.org/package/connect-multiparty#readme)
- [formidable](https://www.npmjs.org/package/formidable#readme)
- [multer](https://www.npmjs.org/package/multer#readme)

Other body parsers you might be interested in:

- [body](https://www.npmjs.org/package/body#readme)
- [co-body](https://www.npmjs.org/package/co-body#readme)

## Installation

```sh
$ npm install body-parser
```

## API

```js
var bodyParser = require('body-parser')
```

### bodyParser.json(options)

Returns middleware that only parses `json`. This parser accepts any Unicode encoding of the body and supports automatic inflation of `gzip` and `deflate` encodings.

The options are:

- `strict` - only parse objects and arrays. (default: `true`)
- `inflate` - if deflated bodies will be inflated. (default: `true`)
- `limit` - maximum request body size. (default: `<100kb>`)
- `reviver` - passed to `JSON.parse()`
- `type` - request content-type to parse (default: `json`)
- `verify` - function to verify body content

The `type` argument is passed directly to the [type-is](https://www.npmjs.org/package/type-is#readme) library. This can be an extension name (like `json`), a mime type (like `application/json`), or a mime time with a wildcard (like `*/json`).

The `verify` argument, if supplied, is called as `verify(req, res, buf, encoding)`, where `buf` is a `Buffer` of the raw request body and `encoding` is the encoding of the request. The parsing can be aborted by throwing an error.

The `reviver` argument is passed directly to `JSON.parse` as the second argument. You can find more information on this argument [in the MDN documentation about JSON.parse](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/JSON/parse#Example.3A_Using_the_reviver_parameter).

### bodyParser.raw(options)

Returns middleware that parses all bodies as a `Buffer`. This parser supports automatic inflation of `gzip` and `deflate` encodings.

The options are:

- `inflate` - if deflated bodies will be inflated. (default: `true`)
- `limit` - maximum request body size. (default: `<100kb>`)
- `type` - request content-type to parse (default: `application/octet-stream`)
- `verify` - function to verify body content

The `type` argument is passed directly to the [type-is](https://www.npmjs.org/package/type-is#readme) library. This can be an extension name (like `bin`), a mime type (like `application/octet-stream`), or a mime time with a wildcard (like `application/*`).

The `verify` argument, if supplied, is called as `verify(req, res, buf, encoding)`, where `buf` is a `Buffer` of the raw request body and `encoding` is the encoding of the request. The parsing can be aborted by throwing an error.

### bodyParser.text(options)

Returns middleware that parses all bodies as a string. This parser supports automatic inflation of `gzip` and `deflate` encodings.

The options are:

- `defaultCharset` - the default charset to parse as, if not specified in content-type. (default: `utf-8`)
- `inflate` - if deflated bodies will be inflated. (default: `true`)
- `limit` - maximum request body size. (default: `<100kb>`)
- `type` - request content-type to parse (default: `text/plain`)
- `verify` - function to verify body content

The `type` argument is passed directly to the [type-is](https://www.npmjs.org/package/type-is#readme) library. This can be an extension name (like `txt`), a mime type (like `text/plain`), or a mime time with a wildcard (like `text/*`).

The `verify` argument, if supplied, is called as `verify(req, res, buf, encoding)`, where `buf` is a `Buffer` of the raw request body and `encoding` is the encoding of the request. The parsing can be aborted by throwing an error.

### bodyParser.urlencoded(options)

Returns middleware that only parses `urlencoded` bodies. This parser accepts only UTF-8 encoding of the body and supports automatic inflation of `gzip` and `deflate` encodings.

The options are:

- `extended` - parse extended syntax with the [qs](https://www.npmjs.org/package/qs#readme) module. (default: `true`)
- `inflate` - if deflated bodies will be inflated. (default: `true`)
- `limit` - maximum request body size. (default: `<100kb>`)
- `parameterLimit` - maximum number of parameters. (default: `1000`)
- `type` - request content-type to parse (default: `urlencoded`)
- `verify` - function to verify body content

The `extended` argument allows to choose between parsing the urlencoded data with the `querystring` library (when `false`) or the `qs` library (when `true`). The "extended" syntax allows for rich objects and arrays to be encoded into the urlencoded format, allowing for a JSON-like experience with urlencoded. For more information, please [see the qs library](https://www.npmjs.org/package/qs#readme).

The `parameterLimit` argument controls the maximum number of parameters that are allowed in the urlencoded data. If a request contains more parameters than this value, a 415 will be returned to the client.

The `type` argument is passed directly to the [type-is](https://www.npmjs.org/package/type-is#readme) library. This can be an extension name (like `urlencoded`), a mime type (like `application/x-www-form-urlencoded`), or a mime time with a wildcard (like `*/x-www-form-urlencoded`).

The `verify` argument, if supplied, is called as `verify(req, res, buf, encoding)`, where `buf` is a `Buffer` of the raw request body and `encoding` is the encoding of the request. The parsing can be aborted by throwing an error.

### req.body

A new `body` object containing the parsed data is populated on the `request` object after the middleware.

## Examples

### express/connect top-level generic

This example demonstrates adding a generic JSON and urlencoded parser as a top-level middleware, which will parse the bodies of all incoming requests. This is the simplest setup.

```js
var express = require('express')
var bodyParser = require('body-parser')

var app = express()

// parse application/x-www-form-urlencoded
app.use(bodyParser.urlencoded({ extended: false }))

// parse application/json
app.use(bodyParser.json())

app.use(function (req, res) {
  res.setHeader('Content-Type', 'text/plain')
  res.write('you posted:\n')
  res.end(JSON.stringify(req.body, null, 2))
})
```

### express route-specific

This example demonstrates adding body parsers specifically to the routes that need them. In general, this is the most recommend way to use body-parser with express.

```js
var express = require('express')
var bodyParser = require('body-parser')

var app = express()

// create application/json parser
var jsonParser = bodyParser.json()

// create application/x-www-form-urlencoded parser
var urlencodedParser = bodyParser.urlencoded({ extended: false })

// POST /login gets urlencoded bodies
app.post('/login', urlencodedParser, function (req, res) {
  if (!req.body) return res.sendStatus(400)
  res.send('welcome, ' + res.body.username)
})

// POST /api/users gets JSON bodies
app.post('/api/users', jsonParser, function (req, res) {
  if (!req.body) return res.sendStatus(400)
  // create user in req.body
})
```

### change content-type for parsers

All the parsers accept a `type` option which allows you to change the `Content-Type` that the middleware will parse.

```js
// parse various different custom JSON types as JSON
app.use(bodyParser.json({ type: 'application/*+json' }))

// parse some custom thing into a Buffer
app.use(bodyParser.raw({ type: 'application/vnd.custom-type' }))

// parse an HTML body into a string
app.use(bodyParser.text({ type: 'text/html' }))
```

## License

[MIT](LICENSE)

[npm-image]: https://img.shields.io/npm/v/body-parser.svg?style=flat
[npm-url]: https://npmjs.org/package/body-parser
[travis-image]: https://img.shields.io/travis/expressjs/body-parser.svg?style=flat
[travis-url]: https://travis-ci.org/expressjs/body-parser
[coveralls-image]: https://img.shields.io/coveralls/expressjs/body-parser.svg?style=flat
[coveralls-url]: https://coveralls.io/r/expressjs/body-parser?branch=master
[downloads-image]: https://img.shields.io/npm/dm/body-parser.svg?style=flat
[downloads-url]: https://npmjs.org/package/body-parser
[gratipay-image]: https://img.shields.io/gratipay/dougwilson.svg?style=flat
[gratipay-url]: https://www.gratipay.com/dougwilson/
