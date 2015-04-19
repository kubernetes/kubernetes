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
goog.require('webdriver.http.CorsClient');
goog.require('webdriver.http.Request');
goog.require('webdriver.test.testutil');

// Alias for readability.
var callbackHelper = webdriver.test.testutil.callbackHelper;

function FakeXhr() {}
FakeXhr.prototype.status = 200;
FakeXhr.prototype.responseText = '';
FakeXhr.prototype.withCredentials = false;
FakeXhr.prototype.open = function() {};
FakeXhr.prototype.send = function() {};
FakeXhr.prototype.setRequestHeader = function() {};

var URL = 'http://localhost:4444/wd/hub';
var REQUEST = new webdriver.http.Request('GET', '/foo');

var control = new goog.testing.MockControl();
var stubs = new goog.testing.PropertyReplacer();
var mockClient, mockXhr;

function setUp() {
  mockClient = control.createStrictMock(webdriver.http.Client);
  mockXhr = control.createStrictMock(FakeXhr);
  mockXhr.status = 200;
  mockXhr.responseText = '';
  mockXhr.withCredentials = false;
  setXhr(mockXhr);
}

function tearDown() {
  control.$tearDown();
  stubs.reset();
}

function setXhr(value) {
  stubs.set(goog.global, 'XMLHttpRequest', function() {
    return value;
  });
  setXdr();
}

function setXdr(opt_value) {
  stubs.set(goog.global, 'XDomainRequest', opt_value);
}

function expectRequest(mockXhr) {
  mockXhr.open('POST', URL + '/xdrpc', true);
  return mockXhr.send(goog.json.serialize({
    'method': REQUEST.method,
    'path': REQUEST.path,
    'data': REQUEST.data
  }));
}

function testDetectsWhenCorsIsAvailable() {
  setXhr(undefined);
  assertFalse(webdriver.http.CorsClient.isAvailable());
  setXhr();
  assertFalse(webdriver.http.CorsClient.isAvailable());
  setXhr({withCredentials: null});
  assertFalse(webdriver.http.CorsClient.isAvailable());
  setXhr({withCredentials: true});
  assertTrue(webdriver.http.CorsClient.isAvailable());
  setXhr();
  setXdr(goog.nullFunction);
  assertTrue(webdriver.http.CorsClient.isAvailable());
}

function testCorsClient_whenUnableToSendARequest() {
  expectRequest(mockXhr).$does(function() {
    mockXhr.onerror();
  });
  control.$replayAll();

  var callback;
  new webdriver.http.CorsClient(URL).send(REQUEST,
      callback = callbackHelper(function(error) {
        assertNotNullNorUndefined(error);
        assertEquals(1, arguments.length);
      }));
  callback.assertCalled();
  control.$verifyAll();
}

function testCorsClient_handlesResponsesWithNoHeaders() {
  expectRequest(mockXhr).$does(function() {
    mockXhr.status = 200;
    mockXhr.responseText = '';
    mockXhr.onload();
  });
  control.$replayAll();

  var callback;
  new webdriver.http.CorsClient(URL).send(REQUEST,
      callback = callbackHelper(function(e, response) {
        assertNull(e);
        assertEquals(200, response.status);
        assertEquals('', response.body);

        webdriver.test.testutil.assertObjectEquals({}, response.headers);
      }));
  callback.assertCalled();
  control.$verifyAll();
}

function testCorsClient_stripsNullCharactersFromResponseBody() {
  expectRequest(mockXhr).$does(function() {
    mockXhr.status = 200;
    mockXhr.responseText = '\x00foo\x00\x00bar\x00';
    mockXhr.onload();
  });
  control.$replayAll();

  var callback;
  new webdriver.http.CorsClient(URL).send(REQUEST,
      callback = callbackHelper(function(e, response) {
        assertNull(e);
        assertEquals(200, response.status);
        assertEquals('foobar', response.body);
        webdriver.test.testutil.assertObjectEquals({}, response.headers);
      }));
  callback.assertCalled();
  control.$verifyAll();
}
