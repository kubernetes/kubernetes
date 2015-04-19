# tiny-lr [![Build Status](https://travis-ci.org/mklabs/tiny-lr.svg?branch=master)](https://travis-ci.org/mklabs/tiny-lr)

This script manages a tiny [LiveReload](http://livereload.com/) server
implementation.

[![NPM](https://nodei.co/npm/tiny-lr.png?compact=true)](https://nodei.co/npm/tiny-lr/)

It exposes an HTTP server and express middleware, with a very basic REST
Api to notify the server of a particular change.

It doesn't have any watch ability, it must be done at the build process or
application level.

Instead, it exposes a very simple API to notify the server that some
changes have been made, then broadcasted to every livereload client
connected.

    # notify a single change
    curl http://localhost:35729/changed?files=style.css

    # notify using a longer path
    curl http://localhost:35729/changed?files=js/app.js

    # notify multiple changes, comma or space delimited
    curl http://localhost:35729/changed?files=index.html,style.css,docs/docco.css

Or you can bulk the information into a POST request, with body as a JSON array of files.

    curl -X POST http://localhost:35729/changed -d '{ "files": ["style.css", "app.js"] }'

    # from a JSON file
    node -pe 'JSON.stringify({ files: ["some.css", "files.css"] })' > files.json
    curl -X POST -d @files.json http://localhost:35729

As for the livereload client, you need to install the browser extension:
http://feedback.livereload.com/knowledgebase/articles/86242-how-do-i-install-and-use-the-browser-extensions-
(**note**: you need to listen on port 35729 to be able to use with your
brower extension)

or add the livereload script tag manually:
http://feedback.livereload.com/knowledgebase/articles/86180-how-do-i-add-the-script-tag-manually-
(and here you can choose whatever port you want)

## Integration

The best way to integrate the runner in your workflow is to add it as a `reload`
step within your build tool.

```js
var tinylr = require('tiny-lr');

// standard LiveReload port
var port = 35729;

// tinylr(opts) => new tinylr.Server(opts);
tinylr().listen(port, function() {
  console.log('... Listening on %s ...', port);
})
```

You can define your own route and listen for specific request:

```js
var server = tinylr();

server.on('GET /myplace', function(req, res) {
  res.write('Mine');
  res.end();
})
```

And stop the server manually:

```js
server.close();
```

This will close any websocket connection established and emit a close event.

### Middleware

To use as a connect / express middleware, tiny-lr needs query /
bodyParser middlewares prior in the stack (to handle POST requests)

Any handled requests ends at the tinylr level, not found and errors are
nexted to the rest of the stack.

```js
var port = process.env.LR_PORT || process.env.PORT || 35729;

var path    = require('path');
var express = require('express');
var tinylr  = require('tiny-lr');
var body    = require('body-parser');

var app = express();

// This binds both express app and tinylr on the same port


app
  .use(body())
  .use(tinylr.middleware({ app: app }))
  .use(express.static(path.resolve('./')))
  .listen(port, function() {
    console.log('listening on %d', port);
  });
```

The port you listen on is important, and tinylr should **always** listen on
the LiveReload standard one: `35729`. Otherwise, you won't be able to rely
on the browser extensions, though you can still use the manual snippet
approach.

You can also start two different servers, one on your app port, the
other listening on the LiveReload port.

### Using grunt

Head over to [https://github.com/gruntjs/grunt-contrib-watch](https://github.com/gruntjs/grunt-contrib-watch#live-reloading)

### Using make

See [make-livereload](https://github.com/mklabs/make-livereload) repo.
This repository defines a bin wrapper you can use and install with:

    npm install make-livereload -g

It bundles the same bin wrapper previously used in tiny-lr repo.

    Usage: tiny-lr [options]

    Options:

      -h, --help     output usage information
      -V, --version  output the version number
      port           -p
      pid            Path to the generated PID file (default: ./tiny-lr.pid)

### Using gulp

See [gulp-livereload](https://github.com/vohof/gulp-livereload) repo.

## Tests

    npm test

---


# TOC
   - [tiny-lr](#tiny-lr)
     - [GET /](#tiny-lr-get-)
     - [GET /changed](#tiny-lr-get-changed)
     - [POST /changed](#tiny-lr-post-changed)
     - [GET /livereload.js](#tiny-lr-get-livereloadjs)
     - [GET /kill](#tiny-lr-get-kill)
<a name="" />

<a name="tiny-lr" />
# tiny-lr
accepts ws clients.

```js
var url = parse(this.request.url);
var server = this.app;

var ws = this.ws = new WebSocket('ws://' + url.host + '/livereload');

ws.onopen = function(event) {
  var hello = {
    command: 'hello',
    protocols: ['http://livereload.com/protocols/official-7']
  };

  ws.send(JSON.stringify(hello));
};

ws.onmessage = function(event) {
  assert.deepEqual(event.data, JSON.stringify({
    command: 'hello',
    protocols: ['http://livereload.com/protocols/official-7'],
    serverName: 'tiny-lr'
  }));

  assert.ok(Object.keys(server.clients).length);
  done();
};
```

properly cleans up established connection on exit.

```js
var ws = this.ws;

ws.onclose = done.bind(null, null);

request(this.server)
  .get('/kill')
  .expect(200, function() {
    console.log('server shutdown');
  });
```

<a name="tiny-lr" />
# tiny-lr
<a name="tiny-lr-get-" />
## GET /
respond with nothing, but respond.

```js
request(this.server)
  .get('/')
  .expect('Content-Type', /json/)
  .expect('{"tinylr":"Welcome","version":"0.0.1"}')
  .expect(200, done);
```

unknown route respond with proper 404 and error message.

```js
request(this.server)
  .get('/whatev')
  .expect('Content-Type', /json/)
  .expect('{"error":"not_found","reason":"no such route"}')
  .expect(404, done);
```

<a name="tiny-lr-get-changed" />
## GET /changed
with no clients, no files.

```js
request(this.server)
  .get('/changed')
  .expect('Content-Type', /json/)
  .expect(/"clients":\[\]/)
  .expect(/"files":\[\]/)
  .expect(200, done);
```

with no clients, some files.

```js
request(this.server)
  .get('/changed?files=gonna.css,test.css,it.css')
  .expect('Content-Type', /json/)
  .expect('{"clients":[],"files":["gonna.css","test.css","it.css"]}')
  .expect(200, done);
```

<a name="tiny-lr-post-changed" />
## POST /changed
with no clients, no files.

```js
request(this.server)
  .post('/changed')
  .expect('Content-Type', /json/)
  .expect(/"clients":\[\]/)
  .expect(/"files":\[\]/)
  .expect(200, done);
```

with no clients, some files.

```js
var data = { clients: [], files: ['cat.css', 'sed.css', 'ack.js'] };

request(this.server)
  .post('/changed')
  .send({ files: data.files })
  .expect('Content-Type', /json/)
  .expect(JSON.stringify(data))
  .expect(200, done);
```

<a name="tiny-lr-get-livereloadjs" />
## GET /livereload.js
respond with livereload script.

```js
request(this.server)
  .get('/livereload.js')
  .expect(/LiveReload/)
  .expect(200, done);
```

<a name="tiny-lr-get-kill" />
## GET /kill
shutdown the server.

```js
var server = this.server;
request(server)
  .get('/kill')
  .expect(200, function(err) {
    if(err) return done(err);
    assert.ok(!server._handle);
    done();
  });
```

## Thanks!

- Tiny-lr is a [LiveReload](http://livereload.com/) implementation. They
  really made frontend editing better for a lot of us. They have a
  [LiveReload App on the Mac App Store](https://itunes.apple.com/us/app/livereload/id482898991)
  you might want to check out.

- To all [contributors](https://github.com/mklabs/tiny-lr/graphs/contributors)

- [@FGRibreau](https://github.com/FGRibreau) / [pid.js
  gist](https://gist.github.com/1846952)) for the background friendly
bin wrapper, used in [make-livereload](https://github.com/mklabs/make-livereload)
