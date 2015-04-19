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
    test = require('../lib/test'),
    assert = require('../testing/assert'),
    Browser = test.Browser,
    Pages = test.Pages;


test.suite(function(env) {
  var browsers = env.browsers;

  var driver;
  beforeEach(function() { driver = env.driver; });

  describe('finding elements', function() {

    test.it(
        'should work after loading multiple pages in a row',
        function() {
          driver.get(Pages.formPage);
          driver.get(Pages.xhtmlTestPage);
          driver.findElement(By.linkText('click me')).click();
          driver.wait(until.titleIs('We Arrive Here'), 5000);
        });

    describe('By.id()', function() {
      test.it('should work', function() {
        driver.get(Pages.xhtmlTestPage);
        driver.findElement(By.id('linkId')).click();
        driver.wait(until.titleIs('We Arrive Here'), 5000);
      });

      test.it('should fail if ID not present on page', function() {
        driver.get(Pages.formPage);
        driver.findElement(By.id('nonExistantButton')).
            then(fail, function(e) {
              assert(e.code).equalTo(error.ErrorCode.NO_SUCH_ELEMENT);
            });
      });

      test.ignore(browsers(Browser.ANDROID)).it(
          'should find multiple elements by ID even though that ' +
              'is malformed HTML',
          function() {
            driver.get(Pages.nestedPage);
            driver.findElements(By.id('2')).then(function(elements) {
              assert(elements.length).equalTo(8);
            });
          });
    });

    describe('By.linkText()', function() {
      test.it('should be able to click on link identified by text', function() {
        driver.get(Pages.xhtmlTestPage);
        driver.findElement(By.linkText('click me')).click();
        driver.wait(until.titleIs('We Arrive Here'), 5000);
      });

      test.it(
        'should be able to find elements by partial link text', function() {
          driver.get(Pages.xhtmlTestPage);
          driver.findElement(By.partialLinkText('ick me')).click();
          driver.wait(until.titleIs('We Arrive Here'), 5000);
        });

      test.it('should work when link text contains equals sign', function() {
        driver.get(Pages.xhtmlTestPage);
        var id = driver.findElement(By.linkText('Link=equalssign')).
            getAttribute('id');
        assert(id).equalTo('linkWithEqualsSign');
      });

      test.it('matches by partial text when containing equals sign',
        function() {
          driver.get(Pages.xhtmlTestPage);
          var id = driver.findElement(By.partialLinkText('Link=')).
              getAttribute('id');
          assert(id).equalTo('linkWithEqualsSign');
        });

      test.it('works when searching for multiple and text contains =',
          function() {
            driver.get(Pages.xhtmlTestPage);
            driver.findElements(By.linkText('Link=equalssign')).
                then(function(elements) {
                  assert(elements.length).equalTo(1);
                  return elements[0].getAttribute('id');
                }).
                then(function(id) {
                  assert(id).equalTo('linkWithEqualsSign');
                });
          });

      test.it(
          'works when searching for multiple with partial text containing =',
          function() {
            driver.get(Pages.xhtmlTestPage);
            driver.findElements(By.partialLinkText('Link=')).
                then(function(elements) {
                  assert(elements.length).equalTo(1);
                  return elements[0].getAttribute('id');
                }).
                then(function(id) {
                  assert(id).equalTo('linkWithEqualsSign');
                });
      });

      test.it('should be able to find multiple exact matches',
          function() {
            driver.get(Pages.xhtmlTestPage);
            driver.findElements(By.linkText('click me')).
                then(function(elements) {
                  assert(elements.length).equalTo(2);
                });
          });

      test.it('should be able to find multiple partial matches',
          function() {
            driver.get(Pages.xhtmlTestPage);
            driver.findElements(By.partialLinkText('ick me')).
                then(function(elements) {
                  assert(elements.length).equalTo(2);
                });
          });

      test.ignore(browsers(Browser.OPERA)).
      it('works on XHTML pages', function() {
        driver.get(test.whereIs('actualXhtmlPage.xhtml'));

        var el = driver.findElement(By.linkText('Foo'));
        assert(el.getText()).equalTo('Foo');
      });
    });

    describe('By.name()', function() {
      test.it('should work', function() {
        driver.get(Pages.formPage);

        var el = driver.findElement(By.name('checky'));
        assert(el.getAttribute('value')).equalTo('furrfu');
      });

      test.it('should find multiple elements with same name', function() {
        driver.get(Pages.nestedPage);
        driver.findElements(By.name('checky')).then(function(elements) {
          assert(elements.length).greaterThan(1);
        });
      });

      test.it(
          'should be able to find elements that do not support name property',
          function() {
            driver.get(Pages.nestedPage);
            driver.findElement(By.name('div1'));
            // Pass if this does not return an error.
          });

      test.it('shoudl be able to find hidden elements by name', function() {
        driver.get(Pages.formPage);
        driver.findElement(By.name('hidden'));
        // Pass if this does not return an error.
      });
    });

    describe('By.className()', function() {
      test.it('should work', function() {
        driver.get(Pages.xhtmlTestPage);

        var el = driver.findElement(By.className('extraDiv'));
        assert(el.getText()).startsWith('Another div starts here.');
      });

      test.it('should work when name is first name among many', function() {
        driver.get(Pages.xhtmlTestPage);

        var el = driver.findElement(By.className('nameA'));
        assert(el.getText()).equalTo('An H2 title');
      });

      test.it('should work when name is last name among many', function() {
        driver.get(Pages.xhtmlTestPage);

        var el = driver.findElement(By.className('nameC'));
        assert(el.getText()).equalTo('An H2 title');
      });

      test.it('should work when name is middle of many', function() {
        driver.get(Pages.xhtmlTestPage);

        var el = driver.findElement(By.className('nameBnoise'));
        assert(el.getText()).equalTo('An H2 title');
      });

      test.it('should work when name surrounded by whitespace', function() {
        driver.get(Pages.xhtmlTestPage);

        var el = driver.findElement(By.className('spaceAround'));
        assert(el.getText()).equalTo('Spaced out');
      });

      test.it('should fail if queried name only partially matches', function() {
        driver.get(Pages.xhtmlTestPage);
        driver.findElement(By.className('nameB')).
            then(fail, function(e) {
              assert(e.code).equalTo(error.ErrorCode.NO_SUCH_ELEMENT);
            });
      });

      test.it('should be able to find multiple matches', function() {
        driver.get(Pages.xhtmlTestPage);
        driver.findElements(By.className('nameC')).then(function(elements) {
          assert(elements.length).greaterThan(1);
        });
      });

      test.it('does not permit compound class names', function() {
        driver.get(Pages.xhtmlTestPage);
        driver.findElement(By.className('a b')).then(fail, pass);
        driver.findElements(By.className('a b')).then(fail, pass);
        function pass() {}
      });
    });

    describe('By.xpath()', function() {
      test.it('should work with multiple matches', function() {
        driver.get(Pages.xhtmlTestPage);
        driver.findElements(By.xpath('//div')).then(function(elements) {
          assert(elements.length).greaterThan(1);
        });
      });

      test.it('should work for selectors using contains keyword', function() {
        driver.get(Pages.nestedPage);
        driver.findElement(By.xpath('//a[contains(., "hello world")]'));
        // Pass if no error.
      });
    });

    describe('By.tagName()', function() {
      test.it('works', function() {
        driver.get(Pages.formPage);

        var el = driver.findElement(By.tagName('input'));
        assert(el.getTagName()).equalTo('input');
      });

      test.it('can find multiple elements', function() {
        driver.get(Pages.formPage);
        driver.findElements(By.tagName('input')).then(function(elements) {
          assert(elements.length).greaterThan(1);
        });
      });
    });

    describe('By.css()', function() {
      test.it('works', function() {
        driver.get(Pages.xhtmlTestPage);
        driver.findElement(By.css('div.content'));
        // Pass if no error.
      });

      test.it('can find multiple elements', function() {
        driver.get(Pages.xhtmlTestPage);
        driver.findElements(By.css('p')).then(function(elements) {
          assert(elements.length).greaterThan(1);
        });
        // Pass if no error.
      });

      test.it(
          'should find first matching element when searching by ' +
              'compound CSS selector',
          function() {
            driver.get(Pages.xhtmlTestPage);
            var el = driver.findElement(By.css('div.extraDiv, div.content'));
            assert(el.getAttribute('class')).equalTo('content');
          });

      test.it('should be able to find multiple elements by compound selector',
          function() {
            driver.get(Pages.xhtmlTestPage);
            driver.findElements(By.css('div.extraDiv, div.content')).
                then(function(elements) {
                  assertClassIs(elements[0], 'content');
                  assertClassIs(elements[1], 'extraDiv');

                  function assertClassIs(el, expected) {
                    assert(el.getAttribute('class')).equalTo(expected);
                  }
                });
          });

      // IE only supports short version option[selected].
      test.ignore(browsers(Browser.IE)).
      it('should be able to find element by boolean attribute', function() {
        driver.get(test.whereIs(
            'locators_tests/boolean_attribute_selected.html'));

        var el = driver.findElement(By.css('option[selected="selected"]'));
        assert(el.getAttribute('value')).equalTo('two');
      });

      test.it(
          'should be able to find element with short ' +
              'boolean attribute selector',
          function() {
            driver.get(test.whereIs(
                'locators_tests/boolean_attribute_selected.html'));

            var el = driver.findElement(By.css('option[selected]'));
            assert(el.getAttribute('value')).equalTo('two');
          });

      test.it(
          'should be able to find element with short boolean attribute ' +
              'selector on HTML4 page',
          function() {
            driver.get(test.whereIs(
                'locators_tests/boolean_attribute_selected_html4.html'));

            var el = driver.findElement(By.css('option[selected]'));
            assert(el.getAttribute('value')).equalTo('two');
          });
    });
  });
});
