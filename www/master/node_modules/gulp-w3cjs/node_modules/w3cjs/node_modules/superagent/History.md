
0.15.7 / 2013-10-19 
==================

 * pin should.js to 1.3.0 due to breaking change in 2.0.x
 * fix browserify regression

0.15.5 / 2013-10-09 
==================

 * add browser field to support browserify
 * fix .field() value number support

0.15.4 / 2013-07-09 
==================

 * node: add a Request#agent() function to set the http Agent to use

0.15.3 / 2013-07-05 
==================

 * fix .pipe() unzipping on more recent nodes. Closes #240
 * fix passing an empty object to .query() no longer appends "?"
 * fix formidable error handling
 * update formidable

0.15.2 / 2013-07-02 
==================

 * fix: emit 'end' when piping.

0.15.1 / 2013-06-26 
==================

 * add try/catch around parseLinks

0.15.0 / 2013-06-25 
==================

 * make `Response#toError()` have a more meaningful `message`

0.14.9 / 2013-06-15 
==================

 * add debug()s to the node client
 * add .abort() method to node client

0.14.8 / 2013-06-13 
==================

 * set .agent = false always
 * remove X-Requested-With. Closes #189

0.14.7 / 2013-06-06 
==================

 * fix unzip error handling

0.14.6 / 2013-05-23 
==================

 * fix HEAD unzip bug

0.14.5 / 2013-05-23 
==================

 * add flag to ensure the callback is __never__ invoked twice

0.14.4 / 2013-05-22 
==================

 * add superagent.js build output
 * update qs
 * update emitter-component
 * revert "add browser field to support browserify" see GH-221

0.14.3 / 2013-05-18 
==================

 * add browser field to support browserify

0.14.2/ 2013-05-07 
==================

  * add host object check to fix serialization of File/Blobs etc as json

0.14.1 / 2013-04-09 
==================

  * update qs

0.14.0 / 2013-04-02 
==================

  * add client-side basic auth
  * fix retaining of .set() header field case

0.13.0 / 2013-03-13 
==================

  * add progress events to client
  * add simple example
  * add res.headers as alias of res.header for browser client
  * add res.get(field) to node/client

0.12.4 / 2013-02-11 
==================

  * fix get content-type even if can't get other headers in firefox. fixes #181

0.12.3 / 2013-02-11 
==================

  * add quick "progress" event support

0.12.2 / 2013-02-04 
==================

  * add test to check if response acts as a readable stream
  * add ReadableStream in the Response prototype.
  * add test to assert correct redirection when the host changes in the location header.
  * add default Accept-Encoding. Closes #155
  * fix req.pipe() return value of original stream for node parity. Closes #171
  * remove the host header when cleaning headers to properly follow the redirection.

0.12.1 / 2013-01-10 
==================

  * add x-domain error handling

0.12.0 / 2013-01-04 
==================

  * add header persistence on redirects

0.11.0 / 2013-01-02 
==================

  * add .error Error object. Closes #156
  * add forcing of res.text removal for FF HEAD responses. Closes #162
  * add reduce component usage. Closes #90
  * move better-assert dep to development deps

0.10.0 / 2012-11-14 
==================

  * add req.timeout(ms) support for the client

0.9.10 / 2012-11-14 
==================

  * fix client-side .query(str) support

0.9.9 / 2012-11-14 
==================

  * add .parse(fn) support
  * fix socket hangup with dates in querystring. Closes #146
  * fix socket hangup "error" event when a callback of arity 2 is provided

0.9.8 / 2012-11-03 
==================

  * add emission of error from `Request#callback()`
  * add a better fix for nodes weird socket hang up error
  * add PUT/POST/PATCH data support to client short-hand functions
  * add .license property to component.json
  * change client portion to build using component(1)
  * fix GET body support [guille]

0.9.7 / 2012-10-19 
==================

  * fix `.buffer()` `res.text` when no parser matches

0.9.6 / 2012-10-17 
==================

  * change: use `this` when `window` is undefined
  * update to new component spec [juliangruber]
  * fix emission of "data" events for compressed responses without encoding. Closes #125

0.9.5 / 2012-10-01 
==================

  * add field name to .attach()
  * add text "parser"
  * refactor isObject()
  * remove wtf isFunction() helper

0.9.4 / 2012-09-20 
==================

  * fix `Buffer` responses [TooTallNate]
  * fix `res.type` when a "type" param is present [TooTallNate]

0.9.3 / 2012-09-18 
==================

  * remove __GET__ `.send()` == `.query()` special-case (__API__ change !!!)

0.9.2 / 2012-09-17 
==================

  * add `.aborted` prop
  * add `.abort()`. Closes #115

0.9.1 / 2012-09-07 
==================

  * add `.forbidden` response property
  * add component.json
  * change emitter-component to 0.0.5
  * fix client-side tests

0.9.0 / 2012-08-28 
==================

  * add `.timeout(ms)`. Closes #17

0.8.2 / 2012-08-28 
==================

  * fix pathname relative redirects. Closes #112

0.8.1 / 2012-08-21 
==================

  * fix redirects when schema is specified

0.8.0 / 2012-08-19 
==================

  * add `res.buffered` flag
  * add buffering of text/*, json and forms only by default. Closes #61
  * add `.buffer(false)` cancellation
  * add cookie jar support [hunterloftis]
  * add agent functionality [hunterloftis]

0.7.0 / 2012-08-03 
==================

  * allow `query()` to be called after the internal `req` has been created [tootallnate]

0.6.0 / 2012-07-17 
==================

  * add `res.send('foo=bar')` default of "application/x-www-form-urlencoded"

0.5.1 / 2012-07-16 
==================

  * add "methods" dep
  * add `.end()` arity check to node callbacks
  * fix unzip support due to weird node internals

0.5.0 / 2012-06-16 
==================

  * Added "Link" response header field parsing, exposing `res.links`

0.4.3 / 2012-06-15 
==================

  * Added 303, 305 and 307 as redirect status codes [slaskis]
  * Fixed passing an object as the url

0.4.2 / 2012-06-02 
==================

  * Added component support
  * Fixed redirect data

0.4.1 / 2012-04-13 
==================

  * Added HTTP PATCH support
  * Fixed: GET / HEAD when following redirects. Closes #86
  * Fixed Content-Length detection for multibyte chars

0.4.0 / 2012-03-04 
==================

  * Added `.head()` method [browser]. Closes #78
  * Added `make test-cov` support
  * Added multipart request support. Closes #11
  * Added all methods that node supports. Closes #71
  * Added "response" event providing a Response object. Closes #28
  * Added `.query(obj)`. Closes #59
  * Added `res.type` (browser). Closes #54
  * Changed: default `res.body` and `res.files` to {}
  * Fixed: port existing query-string fix (browser). Closes #57

0.3.0 / 2012-01-24 
==================

  * Added deflate/gzip support [guillermo]
  * Added `res.type` (Content-Type void of params)
  * Added `res.statusCode` to mirror node
  * Added `res.headers` to mirror node
  * Changed: parsers take callbacks
  * Fixed optional schema support. Closes #49

0.2.0 / 2012-01-05 
==================

  * Added url auth support
  * Added `.auth(username, password)`
  * Added basic auth support [node]. Closes #41
  * Added `make test-docs`
  * Added guillermo's EventEmitter. Closes #16
  * Removed `Request#data()` for SS, renamed to `send()`
  * Removed `Request#data()` from client, renamed to `send()`
  * Fixed array support. [browser]
  * Fixed array support. Closes #35 [node]
  * Fixed `EventEmitter#emit()`

0.1.3 / 2011-10-25 
==================

  * Added error to callback
  * Bumped node dep for 0.5.x

0.1.2 / 2011-09-24 
==================

  * Added markdown documentation
  * Added `request(url[, fn])` support to the client
  * Added `qs` dependency to package.json
  * Added options for `Request#pipe()`
  * Added support for `request(url, callback)`
  * Added `request(url)` as shortcut for `request.get(url)`
  * Added `Request#pipe(stream)`
  * Added inherit from `Stream`
  * Added multipart support
  * Added ssl support (node)
  * Removed Content-Length field from client
  * Fixed buffering, `setEncoding()` to utf8 [reported by stagas]
  * Fixed "end" event when piping

0.1.1 / 2011-08-20 
==================

  * Added `res.redirect` flag (node)
  * Added redirect support (node)
  * Added `Request#redirects(n)` (node)
  * Added `.set(object)` header field support
  * Fixed `Content-Length` support

0.1.0 / 2011-08-09 
==================

  * Added support for multiple calls to `.data()`
  * Added support for `.get(uri, obj)`
  * Added GET `.data()` querystring support
  * Added IE{6,7,8} support [alexyoung]

0.0.1 / 2011-08-05 
==================

  * Initial commit

