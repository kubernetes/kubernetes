# mime-types
[![NPM version](https://badge.fury.io/js/mime-types.svg)](https://badge.fury.io/js/mime-types) [![Build Status](https://travis-ci.org/expressjs/mime-types.svg?branch=master)](https://travis-ci.org/expressjs/mime-types)

The ultimate javascript content-type utility.

### Install

```sh
$ npm install mime-types
```

#### Similar to [node-mime](https://github.com/broofa/node-mime), except:

- __No fallbacks.__ Instead of naively returning the first available type, `mime-types` simply returns `false`, so do `var type = mime.lookup('unrecognized') || 'application/octet-stream'`.
- No `new Mime()` business, so you could do `var lookup = require('mime-types').lookup`.
- Additional mime types are added such as jade and stylus. Feel free to add more!
- Browser support via Browserify and Component by converting lists to JSON files.

Otherwise, the API is compatible.

### Adding Types

If you'd like to add additional types,
simply create a PR adding the type to `custom.json` and
a reference link to the [sources](SOURCES.md).

Do __NOT__ edit `mime.json` or `node.json`.
Those are pulled using `build.js`.
You should only touch `custom.json`.

## API

```js
var mime = require('mime-types')
```

All functions return `false` if input is invalid or not found.

### mime.lookup(path)

Lookup the content-type associated with a file.

```js
mime.lookup('json')           // 'application/json'
mime.lookup('.md')            // 'text/x-markdown'
mime.lookup('file.html')      // 'text/html'
mime.lookup('folder/file.js') // 'application/javascript'

mime.lookup('cats') // false
```

### mime.contentType(type)

Create a full content-type header given a content-type or extension.

```js
mime.contentType('markdown')  // 'text/x-markdown; charset=utf-8'
mime.contentType('file.json') // 'application/json; charset=utf-8'
```

### mime.extension(type)

Get the default extension for a content-type.

```js
mime.extension('application/octet-stream') // 'bin'
```

### mime.charset(type)

Lookup the implied default charset of a content-type.

```js
mime.charset('text/x-markdown') // 'UTF-8'
```

### mime.types[extension] = type

A map of content-types by extension.

### mime.extensions[type] = [extensions]

A map of extensions by content-type.

### mime.define(types)

Globally add definitions.
`types` must be an object of the form:

```js
{
  "<content-type>": [extensions...],
  "<content-type>": [extensions...]
}
```

See the `.json` files in `lib/` for examples.

## License

[MIT](LICENSE)
