// Copyright 2014 Selenium committers
// Copyright 2014 Software Freedom Conservancy
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

goog.provide('webdriver.test.WebDriver.generator.test');
goog.setTestOnly('webdriver.test.WebDriver.generator.test');

goog.require('goog.testing.AsyncTestCase');
goog.require('goog.testing.jsunit');
goog.require('webdriver.Session');
goog.require('webdriver.WebDriver');


var test = goog.testing.AsyncTestCase.createAndInstall(
    'webdriver_generator_test');

var driver;

function setUp() {
  driver = new webdriver.WebDriver(
      new webdriver.Session('test-session', {}),
      new ExplodingExecutor());
}


function testCanUseGeneratorsWithWebDriverCall() {
  test.waitForAsync();

  driver.call(function* () {
    var x = yield webdriver.promise.fulfilled(1);
    var y = yield webdriver.promise.fulfilled(2);
    return x + y;
  }).then(function(value) {
    assertEquals(3, value);
    test.continueTesting();
  });
}


function testCanDefineScopeOnGeneratorCall() {
  test.waitForAsync();

  driver.call(function* () {
    var x = yield webdriver.promise.fulfilled(1);
    return this.name + x;
  }, {name: 'Bob'}).then(function(value) {
    assertEquals('Bob1', value);
    test.continueTesting();
  });
}


function testCanSpecifyArgsOnGeneratorCall() {
  test.waitForAsync();

  driver.call(function* (a, b) {
    var x = yield webdriver.promise.fulfilled(1);
    var y = yield webdriver.promise.fulfilled(2);
    return [x + y, a, b];
  }, null, 'abc', 123).then(function(value) {
    assertArrayEquals([3, 'abc', 123], value);
    test.continueTesting();
  });
}


function testCanUseGeneratorWithWebDriverWait() {
  var values = [];
  driver.wait(function* () {
    yield values.push(1);
    values.push(yield webdriver.promise.delayed(10).then(function() {
      return 2;
    }));
    yield values.push(3);
    return values.length === 6;
  }, 250).then(function() {
    assertArrayEquals([1, 2, 3, 1, 2, 3], values);
    test.continueTesting();
  });
  test.waitForAsync();
}


/**
 * @constructor
 * @implements {webdriver.CommandExecutor}
 */
function ExplodingExecutor() {}


/** @override */
ExplodingExecutor.prototype.execute = function(command, cb) {
  cb(Error('Unsupported operation'));
};
