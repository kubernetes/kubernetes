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

goog.require('goog.testing.PropertyReplacer');
goog.require('goog.testing.jsunit');
goog.require('webdriver.process');
goog.require('webdriver.test.testutil');

var stubs = new goog.testing.PropertyReplacer();

function tearDown() {
  stubs.reset();
}

function initProcess(windowObject) {
  stubs.set(webdriver.process, 'PROCESS_',
      webdriver.process.initBrowserProcess_(windowObject));
}

function testInitializesEnvironmentVariablesFromLocation() {
  initProcess({location: '?a&b=123&c=456&c=789'});
  assertEquals('', webdriver.process.getEnv('a'));
  assertEquals('123', webdriver.process.getEnv('b'));
  assertEquals('["456","789"]', webdriver.process.getEnv('c'));
  assertUndefined(webdriver.process.getEnv('not-there'));
}

function testSettingEnvironmentVariables() {
  initProcess({});
  assertUndefined(webdriver.process.getEnv('foo'));
  webdriver.process.setEnv('foo', 'bar');
  assertEquals('bar', webdriver.process.getEnv('foo'));
}

function testCoercesNewEnvironmentVariablesToAString() {
  initProcess({});
  assertUndefined(webdriver.process.getEnv('foo'));
  webdriver.process.setEnv('foo', goog.nullFunction);
  assertEquals(goog.nullFunction + '', webdriver.process.getEnv('foo'));

  assertUndefined(webdriver.process.getEnv('bar'));
  webdriver.process.setEnv('bar', 123);
  assertEquals('123', webdriver.process.getEnv('bar'));
}

function testCanUnsetEnvironmentVariables() {
  initProcess({});
  assertUndefined(webdriver.process.getEnv('foo'));

  webdriver.process.setEnv('foo', 'one');
  assertEquals('one', webdriver.process.getEnv('foo'));

  webdriver.process.setEnv('foo');
  assertUndefined(webdriver.process.getEnv('foo'));

  webdriver.process.setEnv('foo', 'two');
  assertEquals('two', webdriver.process.getEnv('foo'));

  webdriver.process.setEnv('foo', null);
  assertUndefined(webdriver.process.getEnv('foo'));
}
