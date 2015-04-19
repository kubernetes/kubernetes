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

var assert = require('../testing/assert'),
    test = require('../lib/test'),
    Browser = test.Browser;


test.suite(function(env) {
  var driver;
  beforeEach(function() {
    driver = env.driver;
    driver.switchTo().defaultContent();
  });

  test.it('can set size of the current window', function() {
    changeSizeBy(-20, -20);
  });

  test.it('can set size of the current window from frame', function() {
    driver.get(test.Pages.framesetPage);
    driver.switchTo().frame('fourth');
    changeSizeBy(-20, -20);
  });

  test.it('can set size of the current window from iframe', function() {
    driver.get(test.Pages.iframePage);
    driver.switchTo().frame('iframe1-name');
    changeSizeBy(-20, -20);
  });

  test.it('can set the window position of the current window', function() {
    driver.manage().window().getPosition().then(function(position) {
      driver.manage().window().setSize(640, 480);
      driver.manage().window().setPosition(position.x + 10, position.y + 10);

      // For phantomjs, setPosition is a no-op and the "window" stays at (0, 0)
      if (env.browser === Browser.PHANTOMJS) {
        driver.manage().window().getPosition().then(function(position) {
          assert(position.x).equalTo(0);
          assert(position.y).equalTo(0);
        });
      } else {
        driver.wait(forPositionToBe(position.x + 10, position.y + 10), 1000);
      }
    });
  });

  test.it('can set the window position from a frame', function() {
    driver.get(test.Pages.iframePage);
    driver.switchTo().frame('iframe1-name');
    driver.manage().window().getPosition().then(function(position) {
      driver.manage().window().setSize(640, 480);
      driver.manage().window().setPosition(position.x + 10, position.y + 10);

      // For phantomjs, setPosition is a no-op and the "window" stays at (0, 0)
      if (env.browser === Browser.PHANTOMJS) {
        driver.manage().window().getPosition().then(function(position) {
          assert(position.x).equalTo(0);
          assert(position.y).equalTo(0);
        });
      } else {
        driver.wait(forPositionToBe(position.x + 10, position.y + 10), 1000);
      }
    });
  });

  function changeSizeBy(dx, dy) {
    driver.manage().window().getSize().then(function(size) {
      driver.manage().window().setSize(size.width + dx, size.height + dy);
      driver.wait(forSizeToBe(size.width + dx, size.height + dy), 1000);
    })
  }

  function forSizeToBe(w, h) {
    return function() {
      return driver.manage().window().getSize().then(function(size) {
        return size.width === w && size.height === h;
      });
    };
  }

  function forPositionToBe(x, y) {
    return function() {
      return driver.manage().window().getPosition().then(function(position) {
        return position.x === x &&
            // On OSX, the window height may be bumped down 22px for the top
            // status bar.
           (position.y >= y && position.y <= (y + 22));
      });
    };
  }
});
