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

/**
 * @fileoverview An example of starting multiple WebDriver clients that run
 * in parallel in separate control flows.
 */

var webdriver = require('..'),
    until = webdriver.until;

for (var i = 0; i < 3; i++) {
  (function(n) {
    var flow = new webdriver.promise.ControlFlow()
        .on('uncaughtException', function(e) {
          console.log('uncaughtException in flow %d: %s', n, e);
        });

    var driver = new webdriver.Builder().
        withCapabilities(webdriver.Capabilities.firefox()).
        setControlFlow(flow).  // Comment out this line to see the difference.
        build();

    // Position and resize window so it's easy to see them running together.
    driver.manage().window().setSize(600, 400);
    driver.manage().window().setPosition(300 * i, 400 * i);

    driver.get('http://www.google.com');
    driver.findElement(webdriver.By.name('q')).sendKeys('webdriver');
    driver.findElement(webdriver.By.name('btnG')).click();
    driver.wait(until.titleIs('webdriver - Google Search'), 1000);

    driver.quit();
  })(i);
}

