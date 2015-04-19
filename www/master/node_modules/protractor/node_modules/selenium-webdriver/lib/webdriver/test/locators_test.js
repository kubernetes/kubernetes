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
goog.require('goog.testing.jsunit');
goog.require('webdriver.By');
goog.require('webdriver.Locator');
goog.require('webdriver.Locator.Strategy');
goog.require('webdriver.test.testutil');

// By is exported by webdriver.By, but IDEs don't recognize
// goog.exportSymbol. Explicitly define it here to make the
// IDE stop complaining.
var By = webdriver.By;

var TARGET = 'some-value';

function testCheckLocator() {
  function assertLocatorTypeAndTarget(expectedLocator, locator) {
    assertEquals('Wrong type', expectedLocator.using, locator.using);
    assertEquals('Wrong target', expectedLocator.value, locator.value);
  }


  for (var prop in webdriver.Locator.Strategy) {
    var obj = {};
    obj[prop] = TARGET;
    assertLocatorTypeAndTarget(
        webdriver.Locator.Strategy[prop](TARGET),
        webdriver.Locator.checkLocator(obj));
    assertLocatorTypeAndTarget(
        webdriver.Locator.Strategy[prop](TARGET),
        webdriver.Locator.checkLocator(By[prop](TARGET)));
  }

  assertEquals(
      'Should accept custom locator functions',
      goog.nullFunction,
      webdriver.Locator.checkLocator(goog.nullFunction));
}

function testToString() {
  assertEquals('By.id("foo")', By.id('foo').toString());
  assertEquals('By.className("foo")', By.className('foo').toString());
  assertEquals('By.linkText("foo")', By.linkText('foo').toString());
}
