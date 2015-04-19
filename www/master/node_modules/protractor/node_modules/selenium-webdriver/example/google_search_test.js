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

/**
 * @fileoverview An example test that may be run using Mocha.
 * Usage: mocha -t 10000 selenium-webdriver/example/google_search_test.js
 */

var By = require('..').By,
    until = require('..').until,
    firefox = require('../firefox'),
    test = require('../testing');


test.describe('Google Search', function() {
  var driver;

  test.before(function() {
    driver = new firefox.Driver();
  });

  test.it('should append query to title', function() {
    driver.get('http://www.google.com');
    driver.findElement(By.name('q')).sendKeys('webdriver');
    driver.findElement(By.name('btnG')).click();
    driver.wait(until.titleIs('webdriver - Google Search'), 1000);
  });

  test.after(function() { driver.quit(); });
});
