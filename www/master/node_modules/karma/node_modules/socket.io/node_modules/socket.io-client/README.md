socket.io
=========

#### Sockets for the rest of us

The `socket.io` client is basically a simple HTTP Socket interface implementation.
It looks similar to WebSocket while providing additional features and
leveraging other transports when WebSocket is not supported by the user's
browser.

```js
var socket = io.connect('http://domain.com');
socket.on('connect', function () {
  // socket connected
});
socket.on('custom event', function () {
  // server emitted a custom event
});
socket.on('disconnect', function () {
  // socket disconnected
});
socket.send('hi there');
```

### Recipes

#### Utilizing namespaces (ie: multiple sockets)

If you want to namespace all the messages and events emitted to a particular
endpoint, simply specify it as part of the `connect` uri:

```js
var chat = io.connect('http://localhost/chat');
chat.on('connect', function () {
  // chat socket connected
});

var news = io.connect('/news'); // io.connect auto-detects host
news.on('connect', function () {
  // news socket connected
});
```

#### Emitting custom events

To ease with the creation of applications, you can emit custom events outside
of the global `message` event.

```js
var socket = io.connect();
socket.emit('server custom event', { my: 'data' });
```

#### Forcing disconnection

```js
var socket = io.connect();
socket.on('connect', function () {
  socket.disconnect();
});
```

### Documentation 

#### io#connect

```js
io.connect(uri, [options]);
```

##### Options:

- *resource*

    socket.io

  The resource is what allows the `socket.io` server to identify incoming connections by `socket.io` clients. In other words, any HTTP server can implement socket.io and still serve other normal, non-realtime HTTP requests.

- *transports*

```js
['websocket', 'flashsocket', 'htmlfile', 'xhr-multipart', 'xhr-polling', 'jsonp-polling']
```

  A list of the transports to attempt to utilize (in order of preference).

- *'connect timeout'*

```js
5000
```

  The amount of milliseconds a transport has to create a connection before we consider it timed out.
  
- *'try multiple transports'*

```js
true
```

  A boolean indicating if we should try other transports when the  connectTimeout occurs.
  
- *reconnect*

```js
true
```

  A boolean indicating if we should automatically reconnect if a connection is disconnected. 
  
- *'reconnection delay'*

```js
500
```

  The amount of milliseconds before we try to connect to the server again. We are using a exponential back off algorithm for the following reconnections, on each reconnect attempt this value will get multiplied (500 > 1000 > 2000 > 4000 > 8000).
  

- *'max reconnection attempts'*

```js
10
```

  The amount of attempts should we make using the current transport to connect to the server? After this we will do one final attempt, and re-try with all enabled transport methods before we give up.

##### Properties:

- *options*

  The passed in options combined with the defaults.

- *connected*

  Whether the socket is connected or not.
  
- *connecting*

  Whether the socket is connecting or not.

- *reconnecting*

  Whether we are reconnecting or not.
  
- *transport*  

  The transport instance.

##### Methods:
  
- *connect(λ)*

  Establishes a connection. If λ is supplied as argument, it will be called once the connection is established.
  
- *send(message)*
  
  A string of data to send.
  
- *disconnect*

  Closes the connection.
  
- *on(event, λ)*

  Adds a listener for the event *event*.

- *once(event, λ)*

  Adds a one time listener for the event *event*. The listener is removed after the first time the event is fired.
  
- *removeListener(event, λ)*

  Removes the listener λ for the event *event*.
  
##### Events:

- *connect*

  Fired when the connection is established and the handshake successful.
  
- *connecting(transport_type)*

    Fired when a connection is attempted, passing the transport name.
  
- *connect_failed*

    Fired when the connection timeout occurs after the last connection attempt.
  This only fires if the `connectTimeout` option is set.
  If the `tryTransportsOnConnectTimeout` option is set, this only fires once all
  possible transports have been tried.
  
- *message(message)*
  
  Fired when a message arrives from the server

- *close*

  Fired when the connection is closed. Be careful with using this event, as some transports will fire it even under temporary, expected disconnections (such as XHR-Polling).
  
- *disconnect*

  Fired when the connection is considered disconnected.
  
- *reconnect(transport_type,reconnectionAttempts)*

  Fired when the connection has been re-established. This only fires if the `reconnect` option is set.

- *reconnecting(reconnectionDelay,reconnectionAttempts)*

  Fired when a reconnection is attempted, passing the next delay for the next reconnection.

- *reconnect_failed*

  Fired when all reconnection attempts have failed and we where unsuccessful in reconnecting to the server.  

### Contributors

Guillermo Rauch &lt;guillermo@learnboost.com&gt;

Arnout Kazemier &lt;info@3rd-eden.com&gt;

### License 

(The MIT License)

Copyright (c) 2010 LearnBoost &lt;dev@learnboost.com&gt;

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
