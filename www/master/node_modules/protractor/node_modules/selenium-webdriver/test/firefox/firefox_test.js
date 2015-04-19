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

'use strict';

var path = require('path');

var firefox = require('../../firefox'),
    test = require('../../lib/test'),
    assert = require('../../testing/assert');


var JETPACK_EXTENSION = path.join(__dirname,
    '../../lib/test/data/firefox/jetpack-sample.xpi');
var NORMAL_EXTENSION = path.join(__dirname,
    '../../lib/test/data/firefox/sample.xpi');


test.suite(function(env) {
  env.autoCreateDriver = false;

  describe('firefox', function() {
    describe('Options', function() {
      test.afterEach(function() {
        return env.dispose();
      });

      test.it('can start Firefox with custom preferences', function() {
        var profile = new firefox.Profile();
        profile.setPreference('general.useragent.override', 'foo;bar');

        var options = new firefox.Options().setProfile(profile);

        var driver = env.driver = new firefox.Driver(options);
        driver.get('data:text/html,<html><div>content</div></html>');

        var userAgent = driver.executeScript(
            'return window.navigator.userAgent');
        assert(userAgent).equalTo('foo;bar');
      });

      test.it('can start Firefox with a jetpack extension', function() {
        var profile = new firefox.Profile();
        profile.addExtension(JETPACK_EXTENSION);

        var options = new firefox.Options().setProfile(profile);

        var driver = env.driver = new firefox.Driver(options);
        loadJetpackPage(driver,
            'data:text/html;charset=UTF-8,<html><div>content</div></html>');
        assert(driver.findElement({id: 'jetpack-sample-banner'}).getText())
            .equalTo('Hello, world!');
      });

      test.it('can start Firefox with a normal extension', function() {
        var profile = new firefox.Profile();
        profile.addExtension(NORMAL_EXTENSION);

        var options = new firefox.Options().setProfile(profile);

        var driver = env.driver = new firefox.Driver(options);
        driver.get('data:text/html,<html><div>content</div></html>');
        assert(driver.findElement({id: 'sample-extension-footer'}).getText())
            .equalTo('Goodbye');
      });

      test.it('can start Firefox with multiple extensions', function() {
        var profile = new firefox.Profile();
        profile.addExtension(JETPACK_EXTENSION);
        profile.addExtension(NORMAL_EXTENSION);

        var options = new firefox.Options().setProfile(profile);

        var driver = env.driver = new firefox.Driver(options);

        loadJetpackPage(driver,
            'data:text/html;charset=UTF-8,<html><div>content</div></html>');
        assert(driver.findElement({id: 'jetpack-sample-banner'}).getText())
            .equalTo('Hello, world!');
        assert(driver.findElement({id: 'sample-extension-footer'}).getText())
            .equalTo('Goodbye');
      });

      function loadJetpackPage(driver, url) {
        // On linux the jetpack extension does not always run the first time
        // we load a page. If this happens, just reload the page (a simple
        // refresh doesn't appear to work).
        driver.wait(function() {
          driver.get(url);
          return driver.isElementPresent({id: 'jetpack-sample-banner'});
        }, 3000);
      }
    });
  });
}, {browsers: ['firefox']});
