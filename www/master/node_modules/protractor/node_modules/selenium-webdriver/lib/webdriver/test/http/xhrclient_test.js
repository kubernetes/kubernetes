// Copyright 2014 Software Freedom Conservancy. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

goog.require('goog.json');
goog.require('goog.testing.MockControl');
goog.require('goog.testing.PropertyReplacer');
goog.require('goog.testing.jsunit');
goog.require('webdriver.http.Request');
goog.require('webdriver.http.XhrClient');
goog.require('webdriver.promise');
goog.require('webdriver.test.testutil');

// Alias for readability.
var callbackHelper = webdriver.test.testutil.callbackHelper;

function FakeXhr() {}
FakeXhr.prototype.status = 200;
FakeXhr.prototype.responseText = '';
FakeXhr.prototype.open = function() {};
FakeXhr.prototype.send = function() {};
FakeXhr.prototype.getAllResponseHeaders = function() {};
FakeXhr.prototype.setRequestHeader = function() {};

var URL = 'http://localhost:4444/wd/hub';
var REQUEST = new webdriver.http.Request('GET', '/foo');

var control = new goog.testing.MockControl();
var stubs = new goog.testing.PropertyReplacer();
var mockClient, mockXhr;

function setUp() {
  mockClient = control.createStrictMock(webdriver.http.Client);
  mockXhr = control.createStrictMock(FakeXhr);
  stubs.set(goog.global, 'XMLHttpRequest', function() {
    return mockXhr;
  });
}

function tearDown() {
  control.$tearDown();
  stubs.reset();
}

function setXhr(value) {
  stubs.set(goog.global, 'XMLHttpRequest', value);
}

function expectRequest(mockXhr) {
  mockXhr.open(REQUEST.method, URL + REQUEST.path, true);
  for (var header in REQUEST.headers) {
    mockXhr.setRequestHeader(header, REQUEST.headers[header]);
  }
  return mockXhr.send(goog.json.serialize(REQUEST.data));
}

function testXhrClient_whenUnableToSendARequest() {
  expectRequest(mockXhr).$does(function() {
    mockXhr.onerror();
  });
  control.$replayAll();

  var callback;
  new webdriver.http.XhrClient(URL).send(REQUEST,
      callback = callbackHelper(function(error) {
        assertNotNullNorUndefined(error);
        assertEquals(1, arguments.length);
      }));
  callback.assertCalled();
  control.$verifyAll();
}

function testXhrClient_parsesResponseHeaders_windows() {
  expectRequest(mockXhr).$does(function() {
    mockXhr.status = 200;
    mockXhr.responseText = '';
    mockXhr.onload();
  });
  mockXhr.getAllResponseHeaders().$returns([
    'a:b',
    'c: d',
    'e :f',
    'g : h'
  ].join('\r\n'));
  control.$replayAll();

  var callback;
  new webdriver.http.XhrClient(URL).send(REQUEST,
      callback = callbackHelper(function(e, response) {
        assertNull(e);

        assertEquals(200, response.status);
        assertEquals('', response.body);

        webdriver.test.testutil.assertObjectEquals({
          'a': 'b',
          'c': 'd',
          'e': 'f',
          'g': 'h'
        }, response.headers);
      }));
  callback.assertCalled();
  control.$verifyAll();
}

function testXhrClient_parsesResponseHeaders_unix() {
  expectRequest(mockXhr).$does(function() {
    mockXhr.status = 200;
    mockXhr.responseText = '';
    mockXhr.onload();
  });
  mockXhr.getAllResponseHeaders().$returns([
    'a:b',
    'c: d',
    'e :f',
    'g : h'
  ].join('\n'));
  control.$replayAll();

  var callback;
  new webdriver.http.XhrClient(URL).send(REQUEST,
      callback = callbackHelper(function(e, response) {
        assertNull(e);
        assertEquals(200, response.status);
        assertEquals('', response.body);

        webdriver.test.testutil.assertObjectEquals({
          'a': 'b',
          'c': 'd',
          'e': 'f',
          'g': 'h'
        }, response.headers);
      }));
  callback.assertCalled();
  control.$verifyAll();
}

function testXhrClient_handlesResponsesWithNoHeaders() {
  expectRequest(mockXhr).$does(function() {
    mockXhr.status = 200;
    mockXhr.responseText = '';
    mockXhr.onload();
  });
  mockXhr.getAllResponseHeaders().$returns('');
  control.$replayAll();

  var callback;
  new webdriver.http.XhrClient(URL).send(REQUEST,
      callback = callbackHelper(function(e, response) {
        assertNull(e);
        assertEquals(200, response.status);
        assertEquals('', response.body);

        webdriver.test.testutil.assertObjectEquals({}, response.headers);
      }));
  callback.assertCalled();
  control.$verifyAll();
}

function testXhrClient_stripsNullCharactersFromResponseBody() {
  expectRequest(mockXhr).$does(function() {
    mockXhr.status = 200;
    mockXhr.responseText = '\x00foo\x00\x00bar\x00';
    mockXhr.onload();
  });
  mockXhr.getAllResponseHeaders().$returns('');
  control.$replayAll();

  var callback;
  new webdriver.http.XhrClient(URL).send(REQUEST,
      callback = callbackHelper(function(e, response) {
        assertNull(e);
        assertEquals(200, response.status);
        assertEquals('foobar', response.body);
        webdriver.test.testutil.assertObjectEquals({}, response.headers);
      }));
  callback.assertCalled();
  control.$verifyAll();
}
