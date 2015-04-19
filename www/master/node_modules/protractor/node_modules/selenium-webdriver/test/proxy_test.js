// Copyright 2013 Selenium committers
// Copyright 2013 Software Freedom Conservancy
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//     You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

'use strict';

var http = require('http'),
    url = require('url');

var promise = require('..').promise,
    proxy = require('../proxy'),
    assert = require('../testing/assert'),
    test = require('../lib/test'),
    Server = require('../lib/test/httpserver').Server,
    Browser = test.Browser,
    Pages = test.Pages;


test.suite(function(env) {
  env.autoCreateDriver = false;

  function writeResponse(res, body, encoding, contentType) {
    res.writeHead(200, {
      'Content-Length': Buffer.byteLength(body, encoding),
      'Content-Type': contentType
    });
    res.end(body);
  }

  function writePacFile(res) {
    writeResponse(res, [
      'function FindProxyForURL(url, host) {',
      '  if (shExpMatch(url, "' + goodbyeServer.url('*') + '")) {',
      '    return "DIRECT";',
      '  }',
      '  return "PROXY ' + proxyServer.host() + '";',
      '}'
    ].join('\n'), 'ascii', 'application/x-javascript-config');
  }

  var proxyServer = new Server(function(req, res) {
    var pathname = url.parse(req.url).pathname;
    if (pathname === '/proxy.pac') {
      return writePacFile(res);
    }

    writeResponse(res, [
      '<!DOCTYPE html>',
      '<title>Proxy page</title>',
      '<h3>This is the proxy landing page</h3>'
    ].join(''), 'utf8', 'text/html; charset=UTF-8');
  });

  var helloServer = new Server(function(req, res) {
    writeResponse(res, [
      '<!DOCTYPE html>',
      '<title>Hello</title>',
      '<h3>Hello, world!</h3>'
    ].join(''), 'utf8', 'text/html; charset=UTF-8');
  });

  var goodbyeServer = new Server(function(req, res) {
    writeResponse(res, [
      '<!DOCTYPE html>',
      '<title>Goodbye</title>',
      '<h3>Goodbye, world!</h3>'
    ].join(''), 'utf8', 'text/html; charset=UTF-8');
  });

  test.before(proxyServer.start.bind(proxyServer));
  test.before(helloServer.start.bind(helloServer));
  test.before(goodbyeServer.start.bind(helloServer));

  test.after(proxyServer.stop.bind(proxyServer));
  test.after(helloServer.stop.bind(helloServer));
  test.after(goodbyeServer.stop.bind(goodbyeServer));

  test.afterEach(env.dispose.bind(env));

  test.ignore(env.browsers(Browser.SAFARI)).  // Proxy support not implemented.
  describe('manual proxy settings', function() {
    // phantomjs 1.9.1 in webdriver mode does not appear to respect proxy
    // settings.
    test.ignore(env.browsers(Browser.PHANTOMJS)).
    it('can configure HTTP proxy host', function() {
      var driver = env.builder().
          setProxy(proxy.manual({
            http: proxyServer.host()
          })).
          build();

      driver.get(helloServer.url());
      assert(driver.getTitle()).equalTo('Proxy page');
      assert(driver.findElement({tagName: 'h3'}).getText()).
          equalTo('This is the proxy landing page');
    });

    // PhantomJS does not support bypassing the proxy for individual hosts.
    test.ignore(env.browsers(Browser.PHANTOMJS)).
    it('can bypass proxy for specific hosts', function() {
      var driver = env.builder().
          setProxy(proxy.manual({
            http: proxyServer.host(),
            bypass: helloServer.host()
          })).
          build();

      driver.get(helloServer.url());
      assert(driver.getTitle()).equalTo('Hello');
      assert(driver.findElement({tagName: 'h3'}).getText()).
          equalTo('Hello, world!');

      driver.get(goodbyeServer.url());
      assert(driver.getTitle()).equalTo('Proxy page');
      assert(driver.findElement({tagName: 'h3'}).getText()).
          equalTo('This is the proxy landing page');
    });

    // TODO: test ftp and https proxies.
  });

  // PhantomJS does not support PAC file proxy configuration.
  // Safari does not support proxies.
  test.ignore(env.browsers(Browser.PHANTOMJS, Browser.SAFARI)).
  describe('pac proxy settings', function() {
    test.it('can configure proxy through PAC file', function() {
      var driver = env.builder().
          setProxy(proxy.pac(proxyServer.url('/proxy.pac'))).
          build();

      driver.get(helloServer.url());
      assert(driver.getTitle()).equalTo('Proxy page');
      assert(driver.findElement({tagName: 'h3'}).getText()).
          equalTo('This is the proxy landing page');

      driver.get(goodbyeServer.url());
      assert(driver.getTitle()).equalTo('Goodbye');
      assert(driver.findElement({tagName: 'h3'}).getText()).
          equalTo('Goodbye, world!');
    });
  });

  // TODO: figure out how to test direct and system proxy settings.
  describe.skip('direct proxy settings', function() {});
  describe.skip('system proxy settings', function() {});
});
