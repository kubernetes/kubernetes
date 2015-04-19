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

goog.require('goog.testing.MockControl');
goog.require('goog.testing.PropertyReplacer');
goog.require('goog.testing.jsunit');
goog.require('webdriver.testing.Client');

var FAKE_WINDOW = {
  location: {
    pathname: '/foo/bar/baz'
  }
};

function FakeXhr() {}
FakeXhr.prototype.open = goog.nullFunction;
FakeXhr.prototype.send = goog.nullFunction;

var stubs = new goog.testing.PropertyReplacer;
var control = new goog.testing.MockControl;
var mockXhr;
var client;

function setUp() {
  client = new webdriver.testing.Client(FAKE_WINDOW);
  mockXhr = control.createStrictMock(FakeXhr);
  stubs.set(goog.net, 'XmlHttp', function() {
    return mockXhr;
  });
}

function tearDown() {
  stubs.reset();
  control.$tearDown();
}

function expectToSendEvent(type, data) {
  mockXhr.open('POST', '/testevent', true);
  mockXhr.send(goog.json.serialize({
    'id': '/foo/bar/baz',
    'type': type,
    'data': data
  }));
  control.$replayAll();
}

function testSendInitEvent() {
  expectToSendEvent('INIT', {});
  client.sendInitEvent();
  control.$verifyAll();
}

function testSendResultsEvent() {
  expectToSendEvent('RESULTS', {
    'isSuccess': false,
    'report': 'boom'
  });
  client.sendResultsEvent(false, 'boom');
  control.$verifyAll();
}

function testSendScreenshotEvent() {
  expectToSendEvent('SCREENSHOT', {
    'name': 'ss01',
    'data': '12412412asdfasf'
  });
  client.sendScreenshotEvent('12412412asdfasf', 'ss01');
  control.$verifyAll();
}
