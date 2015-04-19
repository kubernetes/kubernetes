# Socket.IO

Socket.IO is a Node.JS project that makes WebSockets and realtime possible in
all browsers. It also enhances WebSockets by providing built-in multiplexing,
horizontal scalability, automatic JSON encoding/decoding, and more.

## How to Install

```bash
npm install socket.io
```

## How to use

First, require `socket.io`:

```js
var io = require('socket.io');
```

Next, attach it to a HTTP/HTTPS server. If you're using the fantastic `express`
web framework:

#### Express 3.x

```js
var app = express()
  , server = require('http').createServer(app)
  , io = io.listen(server);

server.listen(80);

io.sockets.on('connection', function (socket) {
  socket.emit('news', { hello: 'world' });
  socket.on('my other event', function (data) {
    console.log(data);
  });
});
```

#### Express 2.x

```js
var app = express.createServer()
  , io = io.listen(app);

app.listen(80);

io.sockets.on('connection', function (socket) {
  socket.emit('news', { hello: 'world' });
  socket.on('my other event', function (data) {
    console.log(data);
  });
});
```

Finally, load it from the client side code:

```html
<script src="/socket.io/socket.io.js"></script>
<script>
  var socket = io.connect('http://localhost');
  socket.on('news', function (data) {
    console.log(data);
    socket.emit('my other event', { my: 'data' });
  });
</script>
```

For more thorough examples, look at the `examples/` directory.

## Short recipes

### Sending and receiving events.

Socket.IO allows you to emit and receive custom events.
Besides `connect`, `message` and `disconnect`, you can emit custom events:

```js
// note, io.listen(<port>) will create a http server for you
var io = require('socket.io').listen(80);

io.sockets.on('connection', function (socket) {
  io.sockets.emit('this', { will: 'be received by everyone' });

  socket.on('private message', function (from, msg) {
    console.log('I received a private message by ', from, ' saying ', msg);
  });

  socket.on('disconnect', function () {
    io.sockets.emit('user disconnected');
  });
});
```

### Storing data associated to a client

Sometimes it's necessary to store data associated with a client that's
necessary for the duration of the session.

#### Server side

```js
var io = require('socket.io').listen(80);

io.sockets.on('connection', function (socket) {
  socket.on('set nickname', function (name) {
    socket.set('nickname', name, function () { socket.emit('ready'); });
  });

  socket.on('msg', function () {
    socket.get('nickname', function (err, name) {
      console.log('Chat message by ', name);
    });
  });
});
```

#### Client side

```html
<script>
  var socket = io.connect('http://localhost');

  socket.on('connect', function () {
    socket.emit('set nickname', prompt('What is your nickname?'));
    socket.on('ready', function () {
      console.log('Connected !');
      socket.emit('msg', prompt('What is your message?'));
    });
  });
</script>
```

### Restricting yourself to a namespace

If you have control over all the messages and events emitted for a particular
application, using the default `/` namespace works.

If you want to leverage 3rd-party code, or produce code to share with others,
socket.io provides a way of namespacing a `socket`.

This has the benefit of `multiplexing` a single connection. Instead of
socket.io using two `WebSocket` connections, it'll use one.

The following example defines a socket that listens on '/chat' and one for
'/news':

#### Server side

```js
var io = require('socket.io').listen(80);

var chat = io
  .of('/chat')
  .on('connection', function (socket) {
    socket.emit('a message', { that: 'only', '/chat': 'will get' });
    chat.emit('a message', { everyone: 'in', '/chat': 'will get' });
  });

var news = io
  .of('/news');
  .on('connection', function (socket) {
    socket.emit('item', { news: 'item' });
  });
```

#### Client side:

```html
<script>
  var chat = io.connect('http://localhost/chat')
    , news = io.connect('http://localhost/news');

  chat.on('connect', function () {
    chat.emit('hi!');
  });

  news.on('news', function () {
    news.emit('woot');
  });
</script>
```

### Sending volatile messages.

Sometimes certain messages can be dropped. Let's say you have an app that
shows realtime tweets for the keyword `bieber`. 

If a certain client is not ready to receive messages (because of network slowness
or other issues, or because he's connected through long polling and is in the
middle of a request-response cycle), if he doesn't receive ALL the tweets related
to bieber your application won't suffer.

In that case, you might want to send those messages as volatile messages.

#### Server side

```js
var io = require('socket.io').listen(80);

io.sockets.on('connection', function (socket) {
  var tweets = setInterval(function () {
    getBieberTweet(function (tweet) {
      socket.volatile.emit('bieber tweet', tweet);
    });
  }, 100);

  socket.on('disconnect', function () {
    clearInterval(tweets);
  });
});
```

#### Client side

In the client side, messages are received the same way whether they're volatile
or not.

### Getting acknowledgements

Sometimes, you might want to get a callback when the client confirmed the message
reception.

To do this, simply pass a function as the last parameter of `.send` or `.emit`.
What's more, when you use `.emit`, the acknowledgement is done by you, which
means you can also pass data along:

#### Server side

```js
var io = require('socket.io').listen(80);

io.sockets.on('connection', function (socket) {
  socket.on('ferret', function (name, fn) {
    fn('woot');
  });
});
```

#### Client side

```html
<script>
  var socket = io.connect(); // TIP: .connect with no args does auto-discovery
  socket.on('connect', function () { // TIP: you can avoid listening on `connect` and listen on events directly too!
    socket.emit('ferret', 'tobi', function (data) {
      console.log(data); // data will be 'woot'
    });
  });
</script>
```

### Broadcasting messages

To broadcast, simply add a `broadcast` flag to `emit` and `send` method calls.
Broadcasting means sending a message to everyone else except for the socket
that starts it.

#### Server side

```js
var io = require('socket.io').listen(80);

io.sockets.on('connection', function (socket) {
  socket.broadcast.emit('user connected');
  socket.broadcast.json.send({ a: 'message' });
});
```

### Rooms

Sometimes you want to put certain sockets in the same room, so that it's easy
to broadcast to all of them together.

Think of this as built-in channels for sockets. Sockets `join` and `leave`
rooms in each socket.

#### Server side

```js
var io = require('socket.io').listen(80);

io.sockets.on('connection', function (socket) {
  socket.join('justin bieber fans');
  socket.broadcast.to('justin bieber fans').emit('new fan');
  io.sockets.in('rammstein fans').emit('new non-fan');
});
```

### Using it just as a cross-browser WebSocket

If you just want the WebSocket semantics, you can do that too.
Simply leverage `send` and listen on the `message` event:

#### Server side

```js
var io = require('socket.io').listen(80);

io.sockets.on('connection', function (socket) {
  socket.on('message', function () { });
  socket.on('disconnect', function () { });
});
```

#### Client side

```html
<script>
  var socket = io.connect('http://localhost/');
  socket.on('connect', function () {
    socket.send('hi');

    socket.on('message', function (msg) {
      // my msg
    });
  });
</script>
```

### Changing configuration

Configuration in socket.io is TJ-style:

#### Server side

```js
var io = require('socket.io').listen(80);

io.configure(function () {
  io.set('transports', ['websocket', 'flashsocket', 'xhr-polling']);
});

io.configure('development', function () {
  io.set('transports', ['websocket', 'xhr-polling']);
  io.enable('log');
});
```

## License 

(The MIT License)

Copyright (c) 2011 Guillermo Rauch &lt;guillermo@learnboost.com&gt;

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
'Software'), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
