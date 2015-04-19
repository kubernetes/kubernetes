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
 * @fileoverview An example WebDriver script using Harmony generator functions.
 * This requires node v0.11 or newer.
 *
 * Usage: node --harmony-generators \
 *     selenium-webdriver/example/google_search_generator.js
 */

var By = require('..').By,
    firefox = require('../firefox');

var driver = new firefox.Driver();

driver.get('http://www.google.com/ncr');
driver.call(function* () {
  var query = yield driver.findElement(By.name('q'));
  query.sendKeys('webdriver');

  var submit = yield driver.findElement(By.name('btnG'));
  submit.click();
});

driver.wait(function* () {
  var title = yield driver.getTitle();
  return 'webdriver - Google Search' === title;
}, 1000);

driver.quit();
