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

goog.require('bot.Error');
goog.require('bot.ErrorCode');
goog.require('bot.response');
goog.require('goog.testing.AsyncTestCase');
goog.require('goog.testing.jsunit');
goog.require('webdriver.By');
goog.require('webdriver.CommandName');
goog.require('webdriver.WebDriver');
goog.require('webdriver.WebElement');
goog.require('webdriver.WebElementPromise');
goog.require('webdriver.until');

var test = goog.testing.AsyncTestCase.createAndInstall('until_test');

var By = webdriver.By;
var until = webdriver.until;
var CommandName = webdriver.CommandName;

var driver, executor;

var TestExecutor = function() {
  this.handlers_ = {};
};


TestExecutor.prototype.on = function(cmd, handler) {
  this.handlers_[cmd] = handler;
  return this;
};


TestExecutor.prototype.execute = function(cmd, callback) {
  if (!this.handlers_[cmd.getName()]) {
    throw Error('Unsupported command: ' + cmd.getName());
  }
  this.handlers_[cmd.getName()](cmd, callback);
};


function setUp() {
  executor = new TestExecutor();
  driver = new webdriver.WebDriver('session-id', executor);
}


function testUntilAbleToSwitchToFrame_failsFastForNonSwitchErrors() {
  test.waitForAsync();

  var e = Error('boom');
  executor.on(CommandName.SWITCH_TO_FRAME, function(cmd, callback) {
    callback(e);
  });

  driver.wait(until.ableToSwitchToFrame(0), 100).then(fail, function(e2) {
    assertEquals(e, e2);
    test.continueTesting();
  });
}


function testUntilAbleToSwitchToFrame_byIndex() {
  test.waitForAsync();

  executor.on(CommandName.SWITCH_TO_FRAME, function(cmd, callback) {
    callback(null, {status: bot.ErrorCode.SUCCESS});
  });

  driver.wait(until.ableToSwitchToFrame(0), 100).then(function() {
    test.continueTesting();
  }, fail);
}


function testUntilAbleToSwitchToFrame_byWebElement() {
  test.waitForAsync();

  executor.on(CommandName.SWITCH_TO_FRAME, function(cmd, callback) {
    callback(null, {status: bot.ErrorCode.SUCCESS});
  });

  var el = new webdriver.WebElement(driver, {ELEMENT: 1234});
  driver.wait(until.ableToSwitchToFrame(el), 100).then(function() {
    test.continueTesting();
  }, fail);
}


function testUntilAbleToSwitchToFrame_byWebElementPromise() {
  test.waitForAsync();

  executor.on(CommandName.SWITCH_TO_FRAME, function(cmd, callback) {
    callback(null, {status: bot.ErrorCode.SUCCESS});
  });

  var el = new webdriver.WebElementPromise(driver,
      webdriver.promise.fulfilled(
          new webdriver.WebElement(driver, {ELEMENT: 1234})));
  driver.wait(until.ableToSwitchToFrame(el), 100).then(function() {
    test.continueTesting();
  }, fail);
}


function testUntilAbleToSwitchToFrame_byLocator() {
  test.waitForAsync();

  executor.on(CommandName.FIND_ELEMENTS, function(cmd, callback) {
    callback(null, {
      status: bot.ErrorCode.SUCCESS,
      value: [{ELEMENT: 1234}]
    });
  }).on(CommandName.SWITCH_TO_FRAME, function(cmd, callback) {
    callback(null, {status: bot.ErrorCode.SUCCESS});
  });

  driver.wait(until.ableToSwitchToFrame(By.id('foo')), 100).then(function() {
    test.continueTesting();
  }, fail);
}


function testUntilAbleToSwitchToFrame_byLocator_elementNotInitiallyFound() {
  test.waitForAsync();

  var foundResponses = [[], [], [{ELEMENT: 1234}]];
  executor.on(CommandName.FIND_ELEMENTS, function(cmd, callback) {
    callback(null, {
      status: bot.ErrorCode.SUCCESS,
      value: foundResponses.shift()
    });
  }).on(CommandName.SWITCH_TO_FRAME, function(cmd, callback) {
    callback(null, {status: bot.ErrorCode.SUCCESS});
  });

  driver.wait(until.ableToSwitchToFrame(By.id('foo')), 2000).then(function() {
    assertEquals(0, foundResponses.length);
    test.continueTesting();
  }, fail);
}


function testUntilAbleToSwitchToFrame_timesOutIfNeverAbletoSwitchFrames() {
  test.waitForAsync();

  var count = 0;
  executor.on(CommandName.SWITCH_TO_FRAME, function(cmd, callback) {
    count += 1;
    callback(null, {status: bot.ErrorCode.NO_SUCH_FRAME});
  });

  driver.wait(until.ableToSwitchToFrame(0), 100).then(fail, function(e) {
    assertTrue(count > 0);
    assertTrue('Wrong message: ' + e.message, goog.string.startsWith(
        e.message, 'Waiting to be able to switch to frame'));
    test.continueTesting();
  });
}


function testUntilAlertIsPresent_failsFastForNonAlertSwitchErrors() {
  test.waitForAsync();
  driver.wait(until.alertIsPresent(), 100).then(fail, function(e) {
    assertEquals(
        'Unsupported command: ' + CommandName.GET_ALERT_TEXT, e.message);
    test.continueTesting();
  });
}


function testUntilAlertIsPresent() {
  test.waitForAsync();

  var count = 0;
  executor.on(CommandName.GET_ALERT_TEXT, function(cmd, callback) {
    if (count++ < 3) {
      callback(null, {status: bot.ErrorCode.NO_SUCH_ALERT});
    } else {
      callback(null, {status: bot.ErrorCode.SUCCESS});
    }
  }).on(CommandName.DISMISS_ALERT, function(cmd, callback) {
    callback(null, {status: bot.ErrorCode.SUCCESS});
  });

  driver.wait(until.alertIsPresent(), 1000).then(function(alert) {
    assertEquals(4, count);
    return alert.dismiss();
  }).then(function() {
    test.continueTesting();
  });
}


function testUntilTitleIs() {
  test.waitForAsync();

  var titles = ['foo', 'bar', 'baz'];
  executor.on(CommandName.GET_TITLE, function(cmd, callback) {
    callback(null, bot.response.createResponse(titles.shift()));
  });

  driver.wait(until.titleIs('bar'), 3000).then(function() {
    assertArrayEquals(['baz'], titles);
    test.continueTesting();
  });
}


function testUntilTitleContains() {
  test.waitForAsync();

  var titles = ['foo', 'froogle', 'google'];
  executor.on(CommandName.GET_TITLE, function(cmd, callback) {
    callback(null, bot.response.createResponse(titles.shift()));
  });

  driver.wait(until.titleContains('oogle'), 3000).then(function() {
    assertArrayEquals(['google'], titles);
    test.continueTesting();
  });
}


function testUntilTitleMatches() {
  test.waitForAsync();

  var titles = ['foo', 'froogle', 'aaaabc', 'aabbbc', 'google'];
  executor.on(CommandName.GET_TITLE, function(cmd, callback) {
    callback(null, bot.response.createResponse(titles.shift()));
  });

  driver.wait(until.titleMatches(/^a{2,3}b+c$/), 3000).then(function() {
    assertArrayEquals(['google'], titles);
    test.continueTesting();
  });
}


function testUntilElementLocated() {
  test.waitForAsync();

  var responses = [[], [{ELEMENT: 'abc123'}, {ELEMENT: 'foo'}], ['end']];
  executor.on(CommandName.FIND_ELEMENTS, function(cmd, callback) {
    callback(null, bot.response.createResponse(responses.shift()));
  });

  driver.wait(until.elementLocated(By.id('quux')), 2000).then(function(el) {
    return el.getId();
  }).then(function(id) {
    assertArrayEquals([['end']], responses);
    assertEquals('abc123', id['ELEMENT']);
    test.continueTesting();
  });
}


function testUntilElementsLocated() {
  test.waitForAsync();

  var responses = [[], [{ELEMENT: 'abc123'}, {ELEMENT: 'foo'}], ['end']];
  executor.on(CommandName.FIND_ELEMENTS, function(cmd, callback) {
    callback(null, bot.response.createResponse(responses.shift()));
  });

  driver.wait(until.elementsLocated(By.id('quux')), 2000).then(function(els) {
    return webdriver.promise.all(goog.array.map(els, function(el) {
      return el.getId();
    }));
  }).then(function(ids) {
    assertArrayEquals([['end']], responses);
    assertEquals(2, ids.length);
    assertEquals('abc123', ids[0]['ELEMENT']);
    assertEquals('foo', ids[1]['ELEMENT']);
    test.continueTesting();
  });
}


function testUntilStalenessOf() {
  test.waitForAsync();

  var responses = [
    bot.response.createResponse('body'),
    bot.response.createResponse('body'),
    bot.response.createResponse('body'),
    bot.response.createErrorResponse(
        new bot.Error(bot.ErrorCode.STALE_ELEMENT_REFERENCE, 'now stale')),
    ['end']
  ];
  executor.on(CommandName.GET_ELEMENT_TAG_NAME, function(cmd, callback) {
    callback(null, responses.shift());
  });

  var el = new webdriver.WebElement(driver, {ELEMENT:'foo'});
  driver.wait(until.stalenessOf(el), 2000).then(function() {
    assertArrayEquals([['end']], responses);
    test.continueTesting();
  });
}

function runElementStateTest(predicate, command, responses) {
  test.waitForAsync();
  assertTrue(responses.length > 1);

  responses = goog.array.concat(responses, ['end']);
  executor.on(command, function(cmd, callback) {
    callback(null, bot.response.createResponse(responses.shift()));
  });
  driver.wait(predicate, 2000).then(function() {
    assertArrayEquals(['end'], responses);
    test.continueTesting();
  });
}

function testUntilElementIsVisible() {
  var el = new webdriver.WebElement(driver, {ELEMENT:'foo'});
  runElementStateTest(until.elementIsVisible(el),
      CommandName.IS_ELEMENT_DISPLAYED, [false, false, true]);
}


function testUntilElementIsNotVisible() {
  var el = new webdriver.WebElement(driver, {ELEMENT:'foo'});
  runElementStateTest(until.elementIsNotVisible(el),
      CommandName.IS_ELEMENT_DISPLAYED, [true, true, false]);
}


function testUntilElementIsEnabled() {
  var el = new webdriver.WebElement(driver, {ELEMENT:'foo'});
  runElementStateTest(until.elementIsEnabled(el),
      CommandName.IS_ELEMENT_ENABLED, [false, false, true]);
}


function testUntilElementIsDisabled() {
  var el = new webdriver.WebElement(driver, {ELEMENT:'foo'});
  runElementStateTest(until.elementIsDisabled(el),
      CommandName.IS_ELEMENT_ENABLED, [true, true, false]);
}


function testUntilElementIsSelected() {
  var el = new webdriver.WebElement(driver, {ELEMENT:'foo'});
  runElementStateTest(until.elementIsSelected(el),
      CommandName.IS_ELEMENT_SELECTED, [false, false, true]);
}


function testUntilElementIsNotSelected() {
  var el = new webdriver.WebElement(driver, {ELEMENT:'foo'});
  runElementStateTest(until.elementIsNotSelected(el),
      CommandName.IS_ELEMENT_SELECTED, [true, true, false]);
}


function testUntilElementTextIs() {
  var el = new webdriver.WebElement(driver, {ELEMENT:'foo'});
  runElementStateTest(until.elementTextIs(el, 'foobar'),
      CommandName.GET_ELEMENT_TEXT, ['foo', 'fooba', 'foobar']);
}


function testUntilElementTextContains() {
  var el = new webdriver.WebElement(driver, {ELEMENT:'foo'});
  runElementStateTest(until.elementTextContains(el, 'bar'),
      CommandName.GET_ELEMENT_TEXT, ['foo', 'foobaz', 'foobarbaz']);
}


function testUntilElementTextMatches() {
  var el = new webdriver.WebElement(driver, {ELEMENT:'foo'});
  runElementStateTest(until.elementTextMatches(el, /fo+bar{3}/),
      CommandName.GET_ELEMENT_TEXT, ['foo', 'foobar', 'fooobarrr']);
}
