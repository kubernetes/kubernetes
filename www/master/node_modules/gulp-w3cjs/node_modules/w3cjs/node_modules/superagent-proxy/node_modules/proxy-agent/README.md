proxy-agent
===========
### Maps proxy protocols to `http.Agent` implementations

This module provides a function that returns proxying `http.Agent` instances to
use based off of a given proxy URI.

An LRU cache is used so that `http.Agent` instances are transparently re-used for
subsequent HTTP requests to the same proxy server.

The currently implemented protocol mappings are listed in the table below:


| Protocol  | Proxy Agent for `http` requests | Proxy Agent for `https` requests | Example
|:---------:|:-------------------------------:|:--------------------------------:|:--------:
| `http`    | [http-proxy-agent][]            | [https-proxy-agent][]            | `http://proxy-server-over-tcp.com:3128`
| `https`   | [http-proxy-agent][]            | [https-proxy-agent][]            | `https://proxy-server-over-tls.com:3129`
| `socks`   | [socks-proxy-agent][]           | [socks-proxy-agent][]            | `socks://some-socks-proxy.com:9050`


Installation
------------

Install with `npm`:

``` bash
$ npm install proxy-agent
```


Example
-------

``` js
var http = require('http');
var proxy = require('proxy-agent');

// HTTP, HTTPS, or SOCKS proxy to use
var proxyUri = process.env.http_proxy || 'http://168.63.43.102:3128';

var opts = {
  method: 'GET',
  host: 'jsonip.org',
  path: '/',
  // this is the important part!
  agent: proxy(proxyUri)
};

// the rest works just like any other normal HTTP request
http.get(opts, onresponse);

function onresponse (res) {
  console.log(res.statusCode, res.headers);
  res.pipe(process.stdout);
}
```


API
---

### proxy(String uri, Boolean secure) → http.Agent

Returns an `http.Agent` instance based off of the given proxy `uri`, and `secure`
boolean flag. An LRU cache is used, so the same `http.Agent` instance will be
returned if identical args are passed in.


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
