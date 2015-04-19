/*
  node-http-proxy-test.js: http proxy for node.js

  Copyright (c) 2010 Charlie Robbins, Marak Squires and Fedor Indutny

  Permission is hereby granted, free of charge, to any person obtaining
  a copy of this software and associated documentation files (the
  "Software"), to deal in the Software without restriction, including
  without limitation the rights to use, copy, modify, merge, publish,
  distribute, sublicense, and/or sell copies of the Software, and to
  permit persons to whom the Software is furnished to do so, subject to
  the following conditions:

  The above copyright notice and this permission notice shall be
  included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/
 
var assert = require('assert'),
    fs = require('fs'),
    path = require('path'),
    async = require('async'),
    request = require('request'),
    vows = require('vows'),
    macros = require('../macros'),
    helpers = require('../helpers');

var routeFile = path.join(__dirname, 'config.json');

vows.describe(helpers.describe()).addBatch({
  "With a valid target server": {
    "and no latency": {
      "and no headers": macros.http.assertProxied(),
      "and headers": macros.http.assertProxied({
        request: { headers: { host: 'unknown.com' } }
      }),
      "and request close connection header": macros.http.assertProxied({
        request: { headers: { connection: "close" } },
        outputHeaders: { connection: "close" }
      }),
      "and request keep alive connection header": macros.http.assertProxied({
        request: { headers: { connection: "keep-alive" } },
        outputHeaders: { connection: "keep-alive" }
      }),
      "and response close connection header": macros.http.assertProxied({
        request: { headers: { connection: "" } }, // Must explicitly set to "" because otherwise node will automatically add a "connection: keep-alive" header
        targetHeaders: { connection: "close" },
        outputHeaders: { connection: "close" }
      }),
      "and response keep-alive connection header": macros.http.assertProxied({
        request: { headers: { connection: "" } }, // Must explicitly set to "" because otherwise node will automatically add a "connection: keep-alive" header
        targetHeaders: { connection: "keep-alive" },
        outputHeaders: { connection: "keep-alive" }
      }),
      "and response keep-alive connection header from http 1.0 client": macros.http.assertRawHttpProxied({
        rawRequest: "GET / HTTP/1.0\r\n\r\n",
        targetHeaders: { connection: "keep-alive" },
        match: /connection: close/i
      }),
      "and request keep alive from http 1.0 client": macros.http.assertRawHttpProxied({
        rawRequest: "GET / HTTP/1.0\r\nConnection: Keep-Alive\r\n\r\n",
        targetHeaders: { connection: "keep-alive" },
        match: /connection: keep-alive/i
      }),
      "and no connection header": macros.http.assertProxied({
        request: { headers: { connection: "" } }, // Must explicitly set to "" because otherwise node will automatically add a "connection: keep-alive" header
        outputHeaders: { connection: "keep-alive" }
      }),
      "and forwarding enabled": macros.http.assertForwardProxied()
    },
    "and latency": {
      "and no headers": macros.http.assertProxied({
        latency: 2000
      }),
      "and response headers": macros.http.assertProxied({
        targetHeaders: { "x-testheader": "target" },
        proxyHeaders: { "X-TestHeader": "proxy" },
        outputHeaders: { "x-testheader": "target" },
        latency: 1000
      })
    },
    "and timeout set": macros.http.assertProxied({
      shouldFail: true,
      timeout: 2000,
      requestLatency: 4000
    })
  },
  "With a no valid target server": {
    "and no latency": macros.http.assertInvalidProxy(),
    "and latency": macros.http.assertInvalidProxy({
      latency: 2000
    })
  }
}).export(module);
