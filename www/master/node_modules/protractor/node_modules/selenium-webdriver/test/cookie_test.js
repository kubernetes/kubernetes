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

var assert = require('assert'),
    url = require('url');

var test = require('../lib/test'),
    fileserver = require('../lib/test/fileserver'),
    Browser = test.Browser,
    Pages = test.Pages;


test.suite(function(env) {
  var driver;
  beforeEach(function() { driver = env.driver; });

  test.ignore(env.browsers(Browser.SAFARI)).  // Cookie handling is broken.
  describe('Cookie Management;', function() {

    test.beforeEach(function() {
      driver.get(fileserver.Pages.ajaxyPage);
      driver.manage().deleteAllCookies();
      assertHasCookies();
    });

    test.it('can add new cookies', function() {
      var cookie = createCookieSpec();

      driver.manage().addCookie(cookie.name, cookie.value);
      driver.manage().getCookie(cookie.name).then(function(actual) {
        assert.equal(actual.value, cookie.value);
      });
    });

    test.it('can get all cookies', function() {
      var cookie1 = createCookieSpec();
      var cookie2 = createCookieSpec();

      driver.manage().addCookie(cookie1.name, cookie1.value);
      driver.manage().addCookie(cookie2.name, cookie2.value);

      assertHasCookies(cookie1, cookie2);
    });

    test.ignore(env.browsers(Browser.OPERA)).
    it('only returns cookies visible to the current page', function() {
      var cookie1 = createCookieSpec();
      var cookie2 = createCookieSpec();

      driver.manage().addCookie(cookie1.name, cookie1.value);

      var pageUrl = fileserver.whereIs('page/1');
      driver.get(pageUrl);
      driver.manage().addCookie(
          cookie2.name, cookie2.value, url.parse(pageUrl).pathname);
      assertHasCookies(cookie1, cookie2);

      driver.get(fileserver.Pages.ajaxyPage);
      assertHasCookies(cookie1);

      driver.get(pageUrl);
      assertHasCookies(cookie1, cookie2);
    });

    test.it('can delete all cookies', function() {
      var cookie1 = createCookieSpec();
      var cookie2 = createCookieSpec();

      driver.executeScript(
          'document.cookie = arguments[0] + "=" + arguments[1];' +
          'document.cookie = arguments[2] + "=" + arguments[3];',
          cookie1.name, cookie1.value, cookie2.name, cookie2.value);
      assertHasCookies(cookie1, cookie2);

      driver.manage().deleteAllCookies();
      assertHasCookies();
    });

    test.it('can delete cookies by name', function() {
      var cookie1 = createCookieSpec();
      var cookie2 = createCookieSpec();

      driver.executeScript(
          'document.cookie = arguments[0] + "=" + arguments[1];' +
          'document.cookie = arguments[2] + "=" + arguments[3];',
          cookie1.name, cookie1.value, cookie2.name, cookie2.value);
      assertHasCookies(cookie1, cookie2);

      driver.manage().deleteCookie(cookie1.name);
      assertHasCookies(cookie2);
    });

    test.it('should only delete cookie with exact name', function() {
      var cookie1 = createCookieSpec();
      var cookie2 = createCookieSpec();
      var cookie3 = {name: cookie1.name + 'xx', value: cookie1.value};

      driver.executeScript(
          'document.cookie = arguments[0] + "=" + arguments[1];' +
          'document.cookie = arguments[2] + "=" + arguments[3];' +
          'document.cookie = arguments[4] + "=" + arguments[5];',
          cookie1.name, cookie1.value, cookie2.name, cookie2.value,
          cookie3.name, cookie3.value);
      assertHasCookies(cookie1, cookie2, cookie3);

      driver.manage().deleteCookie(cookie1.name);
      assertHasCookies(cookie2, cookie3);
    });

    test.it('can delete cookies set higher in the path', function() {
      var cookie = createCookieSpec();
      var childUrl = fileserver.whereIs('child/childPage.html');
      var grandchildUrl = fileserver.whereIs(
          'child/grandchild/grandchildPage.html');

      driver.get(childUrl);
      driver.manage().addCookie(cookie.name, cookie.value);
      assertHasCookies(cookie);

      driver.get(grandchildUrl);
      assertHasCookies(cookie);

      driver.manage().deleteCookie(cookie.name);
      assertHasCookies();

      driver.get(childUrl);
      assertHasCookies();
    });

    test.ignore(env.browsers(
        Browser.ANDROID, Browser.FIREFOX, Browser.IE, Browser.OPERA)).
    it('should retain cookie expiry', function() {
      var cookie = createCookieSpec();
      var expirationDelay = 5 * 1000;
      var futureTime = Date.now() + expirationDelay;

      driver.manage().addCookie(
          cookie.name, cookie.value, null, null, false, futureTime);
      driver.manage().getCookie(cookie.name).then(function(actual) {
        assert.equal(actual.value, cookie.value);
        // expiry times are exchanged in seconds since January 1, 1970 UTC.
        assert.equal(actual.expiry, Math.floor(futureTime / 1000));
      });

      driver.sleep(expirationDelay);
      assertHasCookies();
    });
  });

  function createCookieSpec() {
    return {
      name: getRandomString(),
      value: getRandomString()
    };
  }

  function buildCookieMap(cookies) {
    var map = {};
    cookies.forEach(function(cookie) {
      map[cookie.name] = cookie;
    });
    return map;
  }

  function assertHasCookies(var_args) {
    var expected = Array.prototype.slice.call(arguments, 0);
    driver.manage().getCookies().then(function(cookies) {
      assert.equal(cookies.length, expected.length,
          'Wrong # of cookies.' +
          '\n  Expected: ' + JSON.stringify(expected) +
          '\n  Was     : ' + JSON.stringify(cookies));

      var map = buildCookieMap(cookies);
      for (var i = 0; i < expected.length; ++i) {
        assert.equal(expected[i].value, map[expected[i].name].value);
      }
    });
  }

  function getRandomString() {
    var x = 1234567890;
    return Math.floor(Math.random() * x).toString(36);
  }
});
