# Ecstatic [![build status](https://secure.travis-ci.org/jesusabdullah/node-ecstatic.png)](http://travis-ci.org/jesusabdullah/node-ecstatic)

![](http://imgur.com/vhub5.png)

A simple static file server middleware. Use it with a raw http server,
express/connect, or flatiron/union!

# Examples:

## express 3.0.x

``` js
var http = require('http');
var express = require('express');
var ecstatic = require('ecstatic');

var app = express();
app.use(ecstatic({ root: __dirname + '/public' }));
http.createServer(app).listen(8080);

console.log('Listening on :8080');
```

## union

``` js
var union = require('union');
var ecstatic = require('ecstatic');

union.createServer({
  before: [
    ecstatic({ root: __dirname + '/public' }),
  ]
}).listen(8080);

console.log('Listening on :8080');
```

## stock http server

``` js
var http = require('http');
var ecstatic = require('ecstatic');

http.createServer(
  ecstatic({ root: __dirname + '/public' })
).listen(8080);

console.log('Listening on :8080');
```
### fall through
To allow fall through to your custom routes:

```js
ecstatic({ root: __dirname + '/public', handleError: false })
```

# API:

## ecstatic(opts);

Pass ecstatic an options hash, and it will return your middleware!

```js
var opts = {
             root          : __dirname + '/public',
             baseDir       : '/',
             cache         : 3600,
             showDir       : false,
             autoIndex     : false,
             humanReadable : true,
             si            : false,
             defaultExt    : 'html',
             gzip          : false
           }
```

If `opts` is a string, the string is assigned to the root folder and all other
options are set to their defaults.

### `opts.root` 

`opts.root` is the directory you want to serve up.

### `opts.baseDir`

`opts.baseDir` is `/` by default, but can be changed to allow your static files
to be served off a specific route. For example, if `opts.baseDir === "blog"`
and `opts.root = "./public"`, requests for `localhost:8080/blog/index.html` will
resolve to `./public/index.html`.

### `opts.cache`

Customize cache control with `opts.cache` , if it is a number then it will set max-age in seconds.
Other wise it will pass through directly to cache-control. Time defaults to 3600 s (ie, 1 hour).

### `opts.showDir`

Turn **on** directory listings with `opts.showDir === true`. Defaults to **false**.

### `opts.autoIndex`

Serve `/path/index.html` when `/path/` is requested.
Turn **off** autoIndexing with `opts.autoIndex === false`. Defaults to **true**.

### `opts.humanReadable`

If autoIndexing is enabled, add human-readable file sizes. Defaults to **true**.
Aliases are `humanreadable` and `human-readable`.

### `opts.si`

If autoIndexing and humanReadable are enabled, print file sizes with base 1000 instead
of base 1024. Name is inferred from cli options for `ls`. Aliased to `index`, the
equivalent option in Apache.

### `opts.defaultExt`

Turn on default file extensions with `opts.defaultExt`. If `opts.defaultExt` is
true, it will default to `html`. For example if you want a request to `/a-file`
to resolve to `./public/a-file.html`, set this to `true`. If you want
`/a-file` to resolve to `./public/a-file.json` instead, set `opts.defaultExt` to
`json`.

### `opts.gzip`

Set `opts.gzip === true` in order to turn on "gzip mode," wherein ecstatic will
serve `./public/some-file.js.gz` in place of `./public/some-file.js` when the
gzipped version exists and ecstatic determines that the behavior is appropriate.

### `opts.handleError`

Turn **off** handleErrors to allow fall-through with `opts.handleError === false`, Defaults to **true**.

## middleware(req, res, next);

This works more or less as you'd expect.

### ecstatic.showDir(folder);

This returns another middleware which will attempt to show a directory view. Turning on auto-indexing is roughly equivalent to adding this middleware after an ecstatic middleware with autoindexing disabled.

### `ecstatic` command

to start a standalone static http server,
run `npm install -g ecstatic` and then run `ecstatic [dir?] [options] --port PORT`
all options work as above, passed in [optimist](https://github.com/substack/node-optimist) style.
`port` defaults to `8000`. If a `dir` or `--root dir` argument is not passed, ecsatic will
serve the current dir.

# Tests:

    npm test

# License:

MIT.
