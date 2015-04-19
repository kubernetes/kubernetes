agent-base
==========
### Turn a function into an `http.Agent` instance
[![Build Status](https://travis-ci.org/TooTallNate/node-agent-base.png?branch=master)](https://travis-ci.org/TooTallNate/node-agent-base)

This module provides an `http.Agent` generator. That is, you pass it an async
callback function, and it returns a new `http.Agent` instance that will invoke the
given callback function when sending outbound HTTP requests.

#### Some subclasses:

Here's some more interesting uses of `agent-base`. Send a pull request to
list yours!

 * [`http-proxy-agent`][http-proxy-agent]: An HTTP(s) proxy `http.Agent` implementation for HTTP endpoints
 * [`https-proxy-agent`][https-proxy-agent]: An HTTP(s) proxy `http.Agent` implementation for HTTPS endpoints
 * [`socks-proxy-agent`][socks-proxy-agent]: A SOCKS (v4a) proxy `http.Agent` implementation for HTTP and HTTPS


Installation
------------

Install with `npm`:

``` bash
$ npm install agent-base
```


Example
-------

Here's a minimal example that creates a new `net.Socket` connection to the server
for every HTTP request (i.e. the equivalent of `agent: false` option):

``` js
var url = require('url');
var net = require('net');
var http = require('http');
var agent = require('agent-base');

var endpoint = 'http://nodejs.org/api/';
var opts = url.parse(endpoint);

// This is the important part!
opts.agent = agent(function (req, opts, fn) {
  if (!opts.port) opts.port = 80;
  var socket = net.connect(opts);
  fn(null, socket);
});

// Everything else works just like normal...
http.get(opts, function (res) {
  console.log('"response" event!', res.headers);
  res.pipe(process.stdout);
});
```


License
-------

(The MIT License)

Copyright (c) 2013 Nathan Rajlich &lt;nathan@tootallnate.net&gt;

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

[http-proxy-agent]: https://github.com/TooTallNate/node-http-proxy-agent
[https-proxy-agent]: https://github.com/TooTallNate/node-https-proxy-agent
[socks-proxy-agent]: https://github.com/TooTallNate/node-socks-proxy-agent
