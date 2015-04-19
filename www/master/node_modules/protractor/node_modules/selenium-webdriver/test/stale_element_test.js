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

var fail = require('assert').fail;

var By = require('..').By,
    error = require('..').error,
    until = require('..').until,
    assert = require('../testing/assert'),
    test = require('../lib/test'),
    Browser = test.Browser,
    Pages = test.Pages;


test.suite(function(env) {
  var driver;
  beforeEach(function() { driver = env.driver; });

  test.it(
      'dynamically removing elements from the DOM trigger a ' +
          'StaleElementReferenceError',
      function() {
        driver.get(Pages.javascriptPage);

        var toBeDeleted = driver.findElement(By.id('deleted'));
        assert(toBeDeleted.isDisplayed()).isTrue();

        driver.findElement(By.id('delete')).click();
        driver.wait(until.stalenessOf(toBeDeleted), 5000);
      });

  test.it('an element found in a different frame is stale', function() {
    driver.get(Pages.missedJsReferencePage);
    driver.switchTo().frame('inner');
    var el = driver.findElement(By.id('oneline'));
    driver.switchTo().defaultContent();
    el.getText().then(fail, function(e) {
      assert(e.code).equalTo(error.ErrorCode.STALE_ELEMENT_REFERENCE);
    });
  });
});