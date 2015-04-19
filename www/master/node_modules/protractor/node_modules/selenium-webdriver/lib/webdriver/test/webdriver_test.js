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

goog.require('bot.ErrorCode');
goog.require('goog.functions');
goog.require('goog.json');
goog.require('goog.testing.PropertyReplacer');
goog.require('goog.testing.MockControl');
goog.require('goog.testing.jsunit');
goog.require('webdriver.Capabilities');
goog.require('webdriver.Command');
goog.require('webdriver.CommandExecutor');
goog.require('webdriver.CommandName');
goog.require('webdriver.WebDriver');
goog.require('webdriver.Session');
goog.require('webdriver.logging');
goog.require('webdriver.promise');
goog.require('webdriver.promise.ControlFlow');
goog.require('webdriver.promise.Deferred');
goog.require('webdriver.promise.Promise');
goog.require('webdriver.test.testutil');
goog.require('webdriver.testing.promise.FlowTester');

var SESSION_ID = 'test_session_id';

var STUB_DRIVER = {
  controlFlow: goog.nullFunction
};

// Alias some long names that interfere with test readability.
var CName = webdriver.CommandName,
    ECode = bot.ErrorCode,
    STUB_ERROR = webdriver.test.testutil.STUB_ERROR,
    throwStubError = webdriver.test.testutil.throwStubError,
    assertIsStubError = webdriver.test.testutil.assertIsStubError,
    callbackHelper = webdriver.test.testutil.callbackHelper,
    callbackPair = webdriver.test.testutil.callbackPair;

// By is exported by webdriver.By, but IDEs don't recognize
// goog.exportSymbol. Explicitly define it here to make the
// IDE stop complaining.
var By = webdriver.By;

var clock;
var driver;
var flowTester;
var mockControl;
var verifyAll;

function setUp() {
  clock = webdriver.test.testutil.createMockClock();
  flowTester = new webdriver.testing.promise.FlowTester(clock, goog.global);
  mockControl = new goog.testing.MockControl();
  verifyAll = callbackHelper(goog.bind(mockControl.$verifyAll, mockControl));
}


function tearDown() {
  flowTester.dispose();
  verifyAll.assertCalled('Never verified mocks');
  clock.uninstall();
  mockControl.$tearDown();
}


function expectedError(code, message) {
  return function(e) {
    assertEquals('Wrong error message', message, e.message);
    assertEquals('Wrong error code', code, e.code);
  };
}


function createCommandMatcher(commandName, parameters) {
  return new goog.testing.mockmatchers.ArgumentMatcher(function(actual) {
    assertEquals('wrong name', commandName, actual.getName());
    var differences = goog.testing.asserts.findDifferences(
        parameters, actual.getParameters());
    assertNull(
        'Wrong parameters for "' + commandName + '"' +
            '\n    Expected: ' + goog.json.serialize(parameters) +
            '\n    Actual: ' + goog.json.serialize(actual.getParameters()),
        differences);
    return true;
  }, commandName + '(' + goog.json.serialize(parameters) + ')');
}


TestHelper = function() {
  this.executor = mockControl.createStrictMock(webdriver.CommandExecutor);
  this.execute = function() {
    fail('Expectations not set!');
  };
};


TestHelper.expectingFailure = function(opt_errback) {
  var helper = new TestHelper();

  helper.execute = function() {
    flowTester.run();
    flowTester.verifyFailure();
    verifyAll();
    if (opt_errback) {
      opt_errback(flowTester.getFailure());
    }
  };

  return helper;
};


TestHelper.expectingSuccess = function(opt_callback) {
  var helper = new TestHelper();

  helper.execute = function() {
    flowTester.run();
    flowTester.verifySuccess();
    verifyAll();
    if (opt_callback) {
      opt_callback();
    }
  };

  return helper;
};


TestHelper.prototype.expect = function(commandName, opt_parameters) {
  return new TestHelper.Command(this, commandName, opt_parameters);
};


TestHelper.prototype.replayAll = function() {
  mockControl.$replayAll();
  return this;
};


TestHelper.Command = function(testHelper, commandName, opt_parameters) {
  this.helper_ = testHelper;
  this.name_ = commandName;
  this.toDo_ = null;
  this.sessionId_ = SESSION_ID;
  this.withParameters(opt_parameters || {});
};


TestHelper.Command.prototype.withParameters = function(parameters) {
  this.parameters_ = parameters;
  if (this.name_ !== CName.NEW_SESSION) {
    this.parameters_['sessionId'] = this.sessionId_;
  }
  return this;
};


TestHelper.Command.prototype.buildExpectation_ = function() {
  var commandMatcher = createCommandMatcher(this.name_, this.parameters_);
  assertNotNull(this.toDo_);
  this.helper_.executor.
      execute(commandMatcher, goog.testing.mockmatchers.isFunction).
      $does(this.toDo_);
};


TestHelper.Command.prototype.andReturn = function(code, opt_value) {
  this.toDo_ = function(command, callback) {
    callback(null, {
      'status': code,
      'sessionId': {
        'value': SESSION_ID
      },
      'value': goog.isDef(opt_value) ? opt_value : null
    });
  };
  return this;
};


TestHelper.Command.prototype.andReturnSuccess = function(opt_returnValue) {
  return this.andReturn(ECode.SUCCESS, opt_returnValue);
};


TestHelper.Command.prototype.andReturnError = function(errCode, opt_value) {
  return this.andReturn(errCode, opt_value);
};


TestHelper.Command.prototype.replayAll = function() {
  if (!this.toDo_) {
    this.andReturnSuccess(null);
  }
  this.buildExpectation_();
  return this.helper_.replayAll();
};


TestHelper.Command.prototype.expect = function(name, opt_parameters) {
  if (!this.toDo_) {
    this.andReturnSuccess(null);
  }
  this.buildExpectation_();
  return this.helper_.expect(name, opt_parameters);
};


/**
 * @param {!(webdriver.Session|webdriver.promise.Promise)=} opt_session The
 *     session to use.
 * @return {!webdriver.WebDriver} A new driver instance.
 */
TestHelper.prototype.createDriver = function(opt_session) {
  var session = opt_session || new webdriver.Session(SESSION_ID, {});
  return new webdriver.WebDriver(session, this.executor);
};


//////////////////////////////////////////////////////////////////////////////
//
//    Tests
//
//////////////////////////////////////////////////////////////////////////////

function testAttachToSession_sessionIsAvailable() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.DESCRIBE_SESSION).
      withParameters({'sessionId': SESSION_ID}).
      andReturnSuccess({'browserName': 'firefox'}).
      replayAll();

  var callback;
  var driver = webdriver.WebDriver.attachToSession(testHelper.executor,
      SESSION_ID);
  driver.getSession().then(callback = callbackHelper(function(session) {
    webdriver.test.testutil.assertObjectEquals({
      'value':'test_session_id'
    }, session.getId());
    assertEquals('firefox', session.getCapability('browserName'));
  }));
  testHelper.execute();
  callback.assertCalled();
}


function testAttachToSession_failsToGetSessionInfo() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.DESCRIBE_SESSION).
      withParameters({'sessionId': SESSION_ID}).
      andReturnError(ECode.UNKNOWN_ERROR, {'message': 'boom'}).
      replayAll();

  var errback;
  var driver = webdriver.WebDriver.attachToSession(testHelper.executor,
      SESSION_ID);
  driver.getSession().then(null, errback = callbackHelper(function(e) {
    assertEquals(bot.ErrorCode.UNKNOWN_ERROR, e.code);
    assertEquals('boom', e.message);
  }));
  testHelper.execute();
  errback.assertCalled();
}


function testAttachToSession_usesActiveFlowByDefault() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.DESCRIBE_SESSION).
      withParameters({'sessionId': SESSION_ID}).
      andReturnSuccess({}).
      replayAll();

  var driver = webdriver.WebDriver.attachToSession(testHelper.executor,
      SESSION_ID);
  assertEquals(driver.controlFlow(), webdriver.promise.controlFlow());
  testHelper.execute();
}


function testAttachToSession_canAttachInCustomFlow() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.DESCRIBE_SESSION).
      withParameters({'sessionId': SESSION_ID}).
      andReturnSuccess({}).
      replayAll();

  var otherFlow = new webdriver.promise.ControlFlow(goog.global);
  var driver = webdriver.WebDriver.attachToSession(testHelper.executor,
      SESSION_ID, otherFlow);
  assertEquals(otherFlow, driver.controlFlow());
  assertNotEquals(otherFlow, webdriver.promise.controlFlow());
  testHelper.execute();
}


function testCreateSession_happyPathWithCapabilitiesHashObject() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.NEW_SESSION).
      withParameters({
        'desiredCapabilities': {'browserName': 'firefox'}
      }).
      andReturnSuccess({'browserName': 'firefox'}).
      replayAll();

  var callback;
  var driver = webdriver.WebDriver.createSession(testHelper.executor, {
    'browserName': 'firefox'
  });
  driver.getSession().then(callback = callbackHelper(function(session) {
    webdriver.test.testutil.assertObjectEquals({
      'value':'test_session_id'
    }, session.getId());
    assertEquals('firefox', session.getCapability('browserName'));
  }));
  testHelper.execute();
  callback.assertCalled();
}


function testCreateSession_happyPathWithCapabilitiesInstance() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.NEW_SESSION).
      withParameters({
        'desiredCapabilities': {'browserName': 'firefox'}
      }).
      andReturnSuccess({'browserName': 'firefox'}).
      replayAll();

  var callback;
  var driver = webdriver.WebDriver.createSession(
      testHelper.executor, webdriver.Capabilities.firefox());
  driver.getSession().then(callback = callbackHelper(function(session) {
    webdriver.test.testutil.assertObjectEquals({
      'value':'test_session_id'
    }, session.getId());
    assertEquals('firefox', session.getCapability('browserName'));
  }));
  testHelper.execute();
  callback.assertCalled();
}


function testCreateSession_failsToCreateSession() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.NEW_SESSION).
      withParameters({
        'desiredCapabilities': {'browserName': 'firefox'}
      }).
      andReturnError(ECode.UNKNOWN_ERROR, {'message': 'boom'}).
      replayAll();

  var errback;
  var driver = webdriver.WebDriver.createSession(testHelper.executor, {
    'browserName': 'firefox'
  });
  driver.getSession().then(null, errback = callbackHelper(function(e) {
    assertEquals(bot.ErrorCode.UNKNOWN_ERROR, e.code);
    assertEquals('boom', e.message);
  }));
  testHelper.execute();
  errback.assertCalled();
}


function testCreateSession_usesActiveFlowByDefault() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.NEW_SESSION).
      withParameters({'desiredCapabilities': {}}).
      andReturnSuccess({}).
      replayAll();

  var driver = webdriver.WebDriver.createSession(testHelper.executor, {});
  assertEquals(webdriver.promise.controlFlow(), driver.controlFlow());
  testHelper.execute();
}


function testCreateSession_canCreateInCustomFlow() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.NEW_SESSION).
      withParameters({'desiredCapabilities': {}}).
      andReturnSuccess({}).
      replayAll();

  var otherFlow = new webdriver.promise.ControlFlow(goog.global);
  var driver = webdriver.WebDriver.createSession(
      testHelper.executor, {}, otherFlow);
  assertEquals(otherFlow, driver.controlFlow());
  assertNotEquals(otherFlow, webdriver.promise.controlFlow());
  testHelper.execute();
}


function testToWireValue_function() {
  var fn = function() { return 'foo'; };
  var callback;
  webdriver.WebDriver.toWireValue_(fn).
      then(callback = callbackHelper(function(value) {
        assertEquals(fn + '', value);
      }));
  callback.assertCalled();
  verifyAll();  // Expected by tear down.
}


function testToWireValue_simpleObject() {
  var expected = {'sessionId': 'foo'};
  var callback;
  webdriver.WebDriver.toWireValue_({
    'sessionId': new webdriver.Session('foo', {})
  }).then(callback = callbackHelper(function(actual) {
    webdriver.test.testutil.assertObjectEquals(expected, actual);
  }));
  callback.assertCalled();
  verifyAll();  // Expected by tear down.
}


function testToWireValue_nestedObject() {
  var expected = {'sessionId': {'value': 'foo'}};
  var callback;
  webdriver.WebDriver.toWireValue_({
    'sessionId': {
      'value': new webdriver.Session('foo', {})
    }
  }).then(callback = callbackHelper(function(actual) {
    webdriver.test.testutil.assertObjectEquals(expected, actual);
  }));
  callback.assertCalled();
  verifyAll();  // Expected by tear down.
}


function testToWireValue_webElement() {
  var expected = {};
  expected[webdriver.WebElement.ELEMENT_KEY] = 'fefifofum';

  var element = new webdriver.WebElement(STUB_DRIVER, expected);
  var callback;
  webdriver.WebDriver.toWireValue_(element).
      then(callback = callbackHelper(function(actual) {
        webdriver.test.testutil.assertObjectEquals(expected, actual);
      }));
  callback.assertCalled();
  verifyAll();  // Expected by tear down.
}


function testToWireValue_webElementPromise() {
  var expected = {};
  expected[webdriver.WebElement.ELEMENT_KEY] = 'fefifofum';

  var element = new webdriver.WebElement(STUB_DRIVER, expected);
  var elementPromise = new webdriver.WebElementPromise(STUB_DRIVER,
      webdriver.promise.fulfilled(element));
  var callback;
  webdriver.WebDriver.toWireValue_(elementPromise).
      then(callback = callbackHelper(function(actual) {
        webdriver.test.testutil.assertObjectEquals(expected, actual);
      }));
  callback.assertCalled();
  verifyAll();  // Expected by tear down.
}


function testToWireValue_domElement() {
  assertThrows(
      goog.partial(webdriver.WebDriver.toWireValue_, document.body));
  verifyAll();  // Expected by tear down.
}


function testToWireValue_simpleArray() {
  var expected = ['foo'];
  var callback;
  webdriver.WebDriver.toWireValue_([new webdriver.Session('foo', {})]).then(
      callback = callbackHelper(function(actual) {
        assertArrayEquals(expected, actual);
      }));
  callback.assertCalled();
  verifyAll();  // Expected by tear down.
}


function testToWireValue_arrayWithWebElement() {
  var elementJson = {};
  elementJson[webdriver.WebElement.ELEMENT_KEY] = 'fefifofum';

  var element = new webdriver.WebElement(STUB_DRIVER, elementJson);
  var callback;
  webdriver.WebDriver.toWireValue_([element]).
      then(callback = callbackHelper(function(actual) {
        assertTrue(goog.isArray(actual));
        assertEquals(1, actual.length);
        webdriver.test.testutil.assertObjectEquals(elementJson, actual[0]);
      }));
  callback.assertCalled();
  verifyAll();  // Expected by tear down.
}


function testToWireValue_arrayWithWebElementPromise() {
  var elementJson = {};
  elementJson[webdriver.WebElement.ELEMENT_KEY] = 'fefifofum';

  var element = new webdriver.WebElement(STUB_DRIVER, elementJson);
  var elementPromise = new webdriver.WebElementPromise(STUB_DRIVER,
      webdriver.promise.fulfilled(element));

  var callback;
  webdriver.WebDriver.toWireValue_([elementPromise]).
      then(callback = callbackHelper(function(actual) {
        assertTrue(goog.isArray(actual));
        assertEquals(1, actual.length);
        webdriver.test.testutil.assertObjectEquals(elementJson, actual[0]);
      }));
  callback.assertCalled();
  verifyAll();  // Expected by tear down.
}


function testToWireValue_complexArray() {
  var elementJson = {};
  elementJson[webdriver.WebElement.ELEMENT_KEY] = 'fefifofum';
  var expected = ['abc', 123, true, elementJson, [123, {'foo': 'bar'}]];

  var element = new webdriver.WebElement(STUB_DRIVER, elementJson);
  var input = ['abc', 123, true, element, [123, {'foo': 'bar'}]];
  var callback;
  webdriver.WebDriver.toWireValue_(input).
      then(callback = callbackHelper(function(actual) {
        webdriver.test.testutil.assertObjectEquals(expected, actual);
      }));
  callback.assertCalled();
  verifyAll();  // Expected by tear down.
}


function testToWireValue_arrayWithNestedPromises() {
  var callback;
  webdriver.WebDriver.toWireValue_([
    'abc',
    webdriver.promise.fulfilled([
      123,
     webdriver.promise.fulfilled(true)
    ])
  ]).then(callback = callbackHelper(function(actual) {
    assertEquals(2, actual.length);
    assertEquals('abc', actual[0]);
    assertArrayEquals([123, true], actual[1]);
  }));
  callback.assertCalled();
  verifyAll();  // Expected by tear down.
}


function testToWireValue_complexHash() {
  var elementJson = {};
  elementJson[webdriver.WebElement.ELEMENT_KEY] = 'fefifofum';
  var expected = {
    'script': 'return 1',
    'args': ['abc', 123, true, elementJson, [123, {'foo': 'bar'}]],
    'sessionId': 'foo'
  };

  var element = new webdriver.WebElement(STUB_DRIVER, elementJson);
  var parameters = {
    'script': 'return 1',
    'args':['abc', 123, true, element, [123, {'foo': 'bar'}]],
    'sessionId': new webdriver.Session('foo', {})
  };

  var callback;
  webdriver.WebDriver.toWireValue_(parameters).
      then(callback = callbackHelper(function(actual) {
        webdriver.test.testutil.assertObjectEquals(expected, actual);
      }));
  callback.assertCalled();
  verifyAll();  // Expected by tear down.
}


function testFromWireValue_primitives() {
  assertEquals(1, webdriver.WebDriver.fromWireValue_({}, 1));
  assertEquals('', webdriver.WebDriver.fromWireValue_({}, ''));
  assertEquals(true, webdriver.WebDriver.fromWireValue_({}, true));

  assertUndefined(webdriver.WebDriver.fromWireValue_({}, undefined));
  assertNull(webdriver.WebDriver.fromWireValue_({}, null));

  verifyAll();  // Expected by tear down.
}


function testFromWireValue_webElements() {
  var json = {};
  json[webdriver.WebElement.ELEMENT_KEY] = 'foo';

  var element = webdriver.WebDriver.fromWireValue_(STUB_DRIVER, json);
  assertEquals(STUB_DRIVER, element.getDriver());

  var callback;
  webdriver.promise.when(element.id_, callback = callbackHelper(function(id) {
    webdriver.test.testutil.assertObjectEquals(json, id);
  }));
  callback.assertCalled();

  verifyAll();  // Expected by tear down.
}


function testFromWireValue_simpleObject() {
  var json = {'sessionId': 'foo'};
  var out = webdriver.WebDriver.fromWireValue_({}, json);
  webdriver.test.testutil.assertObjectEquals(json, out);
  verifyAll();  // Expected by tear down.
}


function testFromWireValue_nestedObject() {
  var json = {'foo': {'bar': 123}};
  var out = webdriver.WebDriver.fromWireValue_({}, json);
  webdriver.test.testutil.assertObjectEquals(json, out);
  verifyAll();  // Expected by tear down.
}


function testFromWireValue_array() {
  var json = [{'foo': {'bar': 123}}];
  var out = webdriver.WebDriver.fromWireValue_({}, json);
  webdriver.test.testutil.assertObjectEquals(json, out);
  verifyAll();  // Expected by tear down.
}


function testFromWireValue_passesThroughFunctionProperties() {
  var json = [{'foo': {'bar': 123}, 'func': goog.nullFunction}];
  var out = webdriver.WebDriver.fromWireValue_({}, json);
  webdriver.test.testutil.assertObjectEquals(json, out);
  verifyAll();  // Expected by tear down.
}


function testDoesNotExecuteCommandIfSessionDoesNotResolve() {
  var session = webdriver.promise.rejected(STUB_ERROR);
  var testHelper = TestHelper.
      expectingFailure(assertIsStubError).
      replayAll();
  testHelper.createDriver(session).getTitle();
  testHelper.execute();
}


function testCommandReturnValuesArePassedToFirstCallback() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.GET_TITLE).
      andReturnSuccess('Google Search').
      replayAll();

  var driver = testHelper.createDriver();
  var callback;
  driver.getTitle().then(callback = callbackHelper(function(title) {
    assertEquals('Google Search', title);
  }));

  testHelper.execute();
  callback.assertCalled();
}


function testStopsCommandExecutionWhenAnErrorOccurs() {
  var testHelper = TestHelper.
      expectingFailure(expectedError(ECode.NO_SUCH_WINDOW, 'window not found')).
      expect(CName.SWITCH_TO_WINDOW).
      withParameters({'name': 'foo'}).
      andReturnError(ECode.NO_SUCH_WINDOW, {'message': 'window not found'}).
      replayAll();

  var driver = testHelper.createDriver();
  driver.switchTo().window('foo');
  driver.getTitle();  // mock should blow if this gets executed

  testHelper.execute();
}


function testCanSuppressCommandFailures() {
  var callback;
  var testHelper = TestHelper.
      expectingSuccess(function() {
        callback.assertCalled();
      }).
      expect(CName.SWITCH_TO_WINDOW).
      withParameters({'name': 'foo'}).
      andReturnError(ECode.NO_SUCH_WINDOW, {'message': 'window not found'}).
      expect(CName.GET_TITLE).
      andReturnSuccess('Google Search').
      replayAll();

  var driver = testHelper.createDriver();
  driver.switchTo().window('foo').
      thenCatch(callback = callbackHelper(function(e) {
        assertEquals(ECode.NO_SUCH_WINDOW, e.code);
        assertEquals('window not found', e.message);
        return true;  // suppress expected failure
      }));
  driver.getTitle();

  // The mock will verify that getTitle was executed, which is what we want.
  testHelper.execute();
}


function testErrorsPropagateUpToTheRunningApplication() {
  var testHelper = TestHelper.
      expectingFailure(expectedError(ECode.NO_SUCH_WINDOW, 'window not found')).
      expect(CName.SWITCH_TO_WINDOW).
      withParameters({'name':'foo'}).
      andReturnError(ECode.NO_SUCH_WINDOW, {'message': 'window not found'}).
      replayAll();

  testHelper.createDriver().switchTo().window('foo');

  testHelper.execute();
}


function testErrbacksThatReturnErrorsStillSwitchToCallbackChain() {
  var callback;
  var testHelper = TestHelper.
      expectingSuccess(function() {
        callback.assertCalled();
      }).
      expect(CName.SWITCH_TO_WINDOW).
      withParameters({'name':'foo'}).
      andReturnError(ECode.NO_SUCH_WINDOW, {'message':'window not found'}).
      replayAll();

  var driver = testHelper.createDriver();
  driver.switchTo().window('foo').
      thenCatch(function() { return STUB_ERROR; }).
      then(callback = callbackHelper(assertIsStubError));

  testHelper.execute();
}


function testErrbacksThrownCanOverrideOriginalError() {
  var testHelper = TestHelper.
      expectingFailure(assertIsStubError).
      expect(CName.SWITCH_TO_WINDOW, {'name': 'foo'}).
      andReturnError(ECode.NO_SUCH_WINDOW, {'message':'window not found'}).
      replayAll();

  var driver = testHelper.createDriver();
  driver.switchTo().window('foo').thenCatch(throwStubError);

  testHelper.execute();
}


function testCannotScheduleCommandsIfTheSessionIdHasBeenDeleted() {
  var testHelper = TestHelper.expectingSuccess().replayAll();
  var driver = testHelper.createDriver();
  delete driver.session_;
  assertThrows(goog.bind(driver.get, driver, 'http://www.google.com'));
  verifyAll();
}


function testDeletesSessionIdAfterQuitting() {
  var driver;
  var testHelper = TestHelper.
      expectingSuccess(function() {
        assertUndefined('Session ID should have been deleted', driver.session_);
      }).
      expect(CName.QUIT).
      replayAll();

  driver = testHelper.createDriver();
  driver.quit();
  testHelper.execute();
}


function testReportsErrorWhenExecutingCommandsAfterExecutingAQuit() {
  var testHelper = TestHelper.
      expectingFailure(expectedError(undefined,
          'This driver instance does not have a valid session ID ' +
          '(did you call WebDriver.quit()?) and may no longer be used.')).
      expect(CName.QUIT).
      replayAll();

  var driver = testHelper.createDriver();
  driver.quit();
  driver.get('http://www.google.com');
  testHelper.execute();
}


function testCallbackCommandsExecuteBeforeNextCommand() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.GET_CURRENT_URL).
      expect(CName.GET, {'url': 'http://www.google.com'}).
      expect(CName.CLOSE).
      expect(CName.GET_TITLE).
      replayAll();

  var driver = testHelper.createDriver();
  driver.getCurrentUrl().then(function() {
    driver.get('http://www.google.com').then(function() {
      driver.close();
    });
  });
  driver.getTitle();

  testHelper.execute();
}


function testEachCallbackFrameRunsToCompletionBeforeTheNext() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.GET_TITLE).
      expect(CName.GET_CURRENT_URL).
      expect(CName.GET_CURRENT_WINDOW_HANDLE).
      expect(CName.CLOSE).
      expect(CName.QUIT).
      replayAll();

  var driver = testHelper.createDriver();
  driver.getTitle().
      // Everything in this callback...
      then(function() {
        driver.getCurrentUrl();
        driver.getWindowHandle();
      }).
      // ...should execute before everything in this callback.
      then(function() {
        driver.close();
      });
  // This should execute after everything above
  driver.quit();

  testHelper.execute();
}


function testNestedCommandFailuresBubbleUpToGlobalHandlerIfUnsuppressed() {
  var testHelper = TestHelper.
      expectingFailure(expectedError(ECode.NO_SUCH_WINDOW, 'window not found')).
      expect(CName.GET_TITLE).
      expect(CName.SWITCH_TO_WINDOW, {'name': 'foo'}).
      andReturnError(ECode.NO_SUCH_WINDOW, {'message':'window not found'}).
      replayAll();

  var driver = testHelper.createDriver();
  driver.getTitle().
      then(function(){
        driver.switchTo().window('foo');
      });

  testHelper.execute();
}


function testNestedCommandFailuresCanBeSuppressWhenTheyOccur() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.GET_TITLE).
      expect(CName.SWITCH_TO_WINDOW, {'name':'foo'}).
      andReturnError(ECode.NO_SUCH_WINDOW, {'message':'window not found'}).
      expect(CName.CLOSE).
      replayAll();

  var driver = testHelper.createDriver();
  driver.getTitle().
      then(function(){
        driver.switchTo().window('foo').
          thenCatch(goog.functions.TRUE);
      });
  driver.close();
  testHelper.execute();
}


function testNestedCommandFailuresBubbleUpThroughTheFrameStack() {
  var callback;
  var testHelper = TestHelper.
      expectingSuccess(function() {
          callback.assertCalled('Error did not bubble up');
      }).
      expect(CName.GET_TITLE).
      expect(CName.SWITCH_TO_WINDOW, {'name':'foo'}).
      andReturnError(ECode.NO_SUCH_WINDOW, {'message':'window not found'}).
      replayAll();

  var driver = testHelper.createDriver();
  driver.getTitle().
      then(function(){
        return driver.switchTo().window('foo');
      }).
      thenCatch(callback = callbackHelper(function(e) {
        assertEquals(ECode.NO_SUCH_WINDOW, e.code);
        assertEquals('window not found', e.message);
        return true;  // Suppress the error.
      }));

  testHelper.execute();
}


function testNestedCommandFailuresCanBeCaughtAndSuppressed() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.GET_TITLE).
      expect(CName.GET_CURRENT_URL).
      expect(CName.SWITCH_TO_WINDOW, {'name':'foo'}).
      andReturnError(ECode.NO_SUCH_WINDOW, {'message':'window not found'}).
      expect(CName.CLOSE).
      replayAll();

  var driver = testHelper.createDriver();
  driver.getTitle().then(function() {
    driver.getCurrentUrl().
        then(function() {
          return driver.switchTo().window('foo');
        }).
        thenCatch(goog.functions.TRUE);
    driver.close();
  });

  // Let the mock verify everything.
  testHelper.execute();
}


function testReturningADeferredResultFromACallback() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.GET_TITLE).
      expect(CName.GET_CURRENT_URL).
      andReturnSuccess('http://www.google.com').
      replayAll();

  var driver = testHelper.createDriver();
  driver.getTitle().
      then(function() {
        return driver.getCurrentUrl();
      }).
      then(function(value) {
        assertEquals('http://www.google.com', value);
      });
  testHelper.execute();
}


function testReturningADeferredResultFromAnErrbackSuppressesTheError() {
  var count = 0;
  var testHelper = TestHelper.expectingSuccess(function() {
        assertEquals(2, count);
      }).
      expect(CName.SWITCH_TO_WINDOW, {'name':'foo'}).
      andReturnError(ECode.NO_SUCH_WINDOW, {'message':'window not found'}).
      expect(CName.GET_CURRENT_URL).
      andReturnSuccess('http://www.google.com').
      replayAll();

  var driver = testHelper.createDriver();
  driver.switchTo().window('foo').
      thenCatch(function(e) {
        assertEquals(ECode.NO_SUCH_WINDOW, e.code);
        assertEquals('window not found', e.message);
        count += 1;
        return driver.getCurrentUrl();
      }).
      then(function(url) {
        count += 1;
        assertEquals('http://www.google.com', url);
      });
  testHelper.execute();
}


function testExecutingACustomFunctionThatReturnsANonDeferred() {
  var called = false;
  var testHelper = TestHelper.expectingSuccess(function() {
        assertTrue('Callback not called', called);
      }).
      replayAll();

  var driver = testHelper.createDriver();
  driver.call(goog.functions.constant('abc123')).then(function(value) {
    called = true;
    assertEquals('abc123', value);
  });
  testHelper.execute();
}


function testExecutionOrderwithCustomFunctions() {
  var msg = [];
  var testHelper = TestHelper.expectingSuccess(function() {
        assertEquals('cheese is tasty!', msg.join(''));
      }).
      expect(CName.GET_TITLE).andReturnSuccess('cheese ').
      expect(CName.GET_CURRENT_URL).andReturnSuccess('tasty').
      replayAll();

  var driver = testHelper.createDriver();

  var pushMsg = goog.bind(msg.push, msg);
  driver.getTitle().then(pushMsg);
  driver.call(goog.functions.constant('is ')).then(pushMsg);
  driver.getCurrentUrl().then(pushMsg);
  driver.call(goog.functions.constant('!')).then(pushMsg);

  testHelper.execute();
}


function testPassingArgumentsToACustomFunction() {
  var testHelper = TestHelper.expectingSuccess().replayAll();

  var add = callbackHelper(function(a, b) {
    return a + b;
  });
  var driver = testHelper.createDriver();
  driver.call(add, null, 1, 2).
      then(function(value) {
        assertEquals(3, value);
      });
  testHelper.execute();
  add.assertCalled();
}

function testPassingPromisedArgumentsToACustomFunction() {
  var testHelper = TestHelper.expectingSuccess().replayAll();

  var promisedArg = new webdriver.promise.Deferred;
  var add = callbackHelper(function(a, b) {
    return a + b;
  });
  var driver = testHelper.createDriver();
  driver.call(add, null, 1, promisedArg).
      then(function(value) {
        assertEquals(3, value);
      });

  flowTester.turnEventLoop();
  add.assertNotCalled();

  promisedArg.fulfill(2);
  add.assertCalled();
  testHelper.execute();
}

function testPassingArgumentsAndScopeToACustomFunction() {
  function Foo(name) {
    this.name = name;
  }
  Foo.prototype.getName = function() {
    return this.name;
  };
  var foo = new Foo('foo');

  var called = false;
  var testHelper = TestHelper.expectingSuccess(function() {
        assertTrue('Callback not called', called);
      }).
      replayAll();
  var driver = testHelper.createDriver();
  driver.call(foo.getName, foo).then(function(value) {
    assertEquals('foo', value);
    called = true;
  });
  testHelper.execute();
}


function testExecutingACustomFunctionThatThrowsAnError() {
  var called = false;
  var testHelper = TestHelper.expectingSuccess(function() {
        assertTrue('Callback not called', called);
      }).
      replayAll();
  var driver = testHelper.createDriver();
  driver.call(goog.functions.error('bam!')).thenCatch(function(e) {
    assertTrue(e instanceof Error);
    assertEquals('bam!', e.message);
    called = true;
    return true;  // suppress the error.
  });
  testHelper.execute();
}


function testExecutingACustomFunctionThatSchedulesCommands() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.GET_TITLE).
      expect(CName.CLOSE).
      expect(CName.QUIT).
      replayAll();

  var driver = testHelper.createDriver();
  driver.call(function() {
    driver.getTitle();
    driver.close();
  });
  driver.quit();
  testHelper.execute();
}


function testExecutingAFunctionThatReturnsATaskResultAfterSchedulingAnother() {
  var called = false;
  var testHelper = TestHelper.expectingSuccess(function() {
        assertTrue(called);
      }).
      expect(CName.GET_TITLE).
          andReturnSuccess('Google Search').
      expect(CName.CLOSE).
      replayAll();

  var driver = testHelper.createDriver();
  var result = driver.call(function() {
    var title = driver.getTitle();
    driver.close();
    return title;
  });

  result.then(function(title) {
    called = true;
    assertEquals('Google Search', title);
  });

  testHelper.execute();
}


function testExecutingACustomFunctionWhoseNestedCommandFails() {
  var called = false;
  var testHelper = TestHelper.expectingSuccess(function() {
        assertTrue('Callback not called', called);
      }).
      expect(CName.SWITCH_TO_WINDOW, {'name': 'foo'}).
      andReturnError(ECode.NO_SUCH_WINDOW, {'message':'window not found'}).
      replayAll();

  var driver = testHelper.createDriver();
  var result = driver.call(function() {
    return driver.switchTo().window('foo');
  });

  result.thenCatch(function(e) {
    assertEquals(ECode.NO_SUCH_WINDOW, e.code);
    assertEquals('window not found', e.message);
    called = true;
    return true;  // suppress the error.
  });

  testHelper.execute();
}


function testCustomFunctionDoesNotCompleteUntilReturnedPromiseIsResolved() {
  var testHelper = TestHelper.expectingSuccess().replayAll();

  var d = new webdriver.promise.Deferred(),
      stepOne = callbackHelper(function() { return d.promise; }),
      stepTwo = callbackHelper();

  var driver = testHelper.createDriver();
  driver.call(stepOne);
  driver.call(stepTwo);

  flowTester.turnEventLoop();
  stepOne.assertCalled();
  stepTwo.assertNotCalled();

  flowTester.turnEventLoop();
  stepOne.assertCalled();
  stepTwo.assertNotCalled();

  d.fulfill();
  testHelper.execute();
  stepTwo.assertCalled();
}


function testNestedFunctionCommandExecutionOrder() {
  var msg = [];
  var testHelper = TestHelper.expectingSuccess(function() {
        assertEquals('acefdb', msg.join(''));
      }).
      replayAll();

  var driver = testHelper.createDriver();
  driver.call(msg.push, msg, 'a');
  driver.call(function() {
    driver.call(msg.push, msg, 'c');
    driver.call(function() {
      driver.call(msg.push, msg, 'e');
      driver.call(msg.push, msg, 'f');
    });
    driver.call(msg.push, msg, 'd');
  });
  driver.call(msg.push, msg, 'b');
  testHelper.execute();
}


function testExecutingNestedFunctionCommands() {
  var msg = [];
  var testHelper = TestHelper.expectingSuccess(function() {
        assertEquals('cheese is tasty!', msg.join(''));
      }).
      replayAll();
  var driver = testHelper.createDriver();
  var pushMsg = goog.bind(msg.push, msg);
  driver.call(goog.functions.constant('cheese ')).then(pushMsg);
  driver.call(function() {
    driver.call(goog.functions.constant('is ')).then(pushMsg);
    driver.call(goog.functions.constant('tasty')).then(pushMsg);
  });
  driver.call(goog.functions.constant('!')).then(pushMsg);
  testHelper.execute();
}


function testReturnValuesFromNestedFunctionCommands() {
  var count = 0;
  var testHelper = TestHelper.expectingSuccess(function() {
        assertEquals('not called', 1, count);
      }).
      replayAll();
  var driver = testHelper.createDriver();
  var result = driver.call(function() {
    return driver.call(function() {
      return driver.call(goog.functions.constant('foobar'));
    });
  });
  result.then(function(value) {
    assertEquals('foobar', value);
    count += 1;
  });
  testHelper.execute();
}


function testExecutingANormalCommandAfterNestedCommandsThatReturnsAnAction() {
  var msg = [];
  var testHelper = TestHelper.expectingSuccess(function() {
        assertEquals('ab', msg.join(''));
      }).
      expect(CName.CLOSE).
      replayAll();
  var driver = testHelper.createDriver();
  driver.call(function() {
    return driver.call(function() {
      msg.push('a');
      return driver.call(goog.functions.constant('foobar'));
    });
  });
  driver.close().then(function() {
    msg.push('b');
  });

  testHelper.execute();
}


function testNestedCommandErrorsBubbleUp() {
  var testHelper = TestHelper.
      expectingFailure(expectedError(undefined, 'bam!')).
      replayAll();
  var driver = testHelper.createDriver();
  driver.call(function() {
    return driver.call(function() {
      return driver.call(goog.functions.error('bam!'));
    });
  });
  testHelper.execute();
}


function testExecutingNestedCustomFunctionsThatSchedulesCommands() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.GET_TITLE).
      expect(CName.CLOSE).
      replayAll();

  var driver = testHelper.createDriver();
  driver.call(function() {
    driver.call(function() {
      driver.getTitle();
    });
    driver.close();
  });
  testHelper.execute();
}


function testExecutingACustomFunctionThatReturnsADeferredAction() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.GET_TITLE).andReturnSuccess('Google').
      replayAll();

  var driver = testHelper.createDriver();
  driver.call(function() {
    return driver.getTitle();
  }).then(function(title) {
    assertEquals('Google', title);
  });

  testHelper.execute();
}

function testWebElementPromise_resolvesWhenUnderlyingElementDoes() {
  var el = new webdriver.WebElement(STUB_DRIVER, {'ELEMENT': 'foo'});
  var d = webdriver.promise.defer();
  var callback;
  new webdriver.WebElementPromise(STUB_DRIVER, d.promise).then(
      callback = callbackHelper(function(e) {
        assertEquals(e, el);
      }));
  callback.assertNotCalled();
  d.fulfill(el);
  callback.assertCalled();
  verifyAll();  // Make tearDown happy.
}

function testWebElement_resolvesBeforeCallbacksOnWireValueTrigger() {
  var el = new webdriver.promise.Deferred();

  var callback, idCallback;
  var element = new webdriver.WebElementPromise(STUB_DRIVER, el.promise);
  var messages = [];

  webdriver.promise.when(element, function() {
    messages.push('element resolved');
  });

  webdriver.promise.when(element.getId(), function() {
    messages.push('wire value resolved');
  });

  assertArrayEquals([], messages);
  el.fulfill(new webdriver.WebElement(STUB_DRIVER, {'ELEMENT': 'foo'}));
  assertArrayEquals([
    'element resolved',
    'wire value resolved'
  ], messages);
  verifyAll();  // Make tearDown happy.
}

function testWebElement_isRejectedIfUnderlyingIdIsRejected() {
  var id = new webdriver.promise.Deferred();

  var callback, errback;
  var element = new webdriver.WebElementPromise(STUB_DRIVER, id.promise);

  webdriver.promise.when(element,
      callback = callbackHelper(),
      errback = callbackHelper(assertIsStubError));

  callback.assertNotCalled();
  errback.assertNotCalled();

  id.reject(STUB_ERROR);

  callback.assertNotCalled();
  errback.assertCalled();
  verifyAll();  // Make tearDown happy.
}


function testExecuteScript_nullReturnValue() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.EXECUTE_SCRIPT).
          withParameters({
            'script': 'return document.body;',
            'args': []
          }).
          andReturnSuccess(null).
      replayAll();

  var driver = testHelper.createDriver();
  var callback;
  driver.executeScript('return document.body;').
      then(callback = callbackHelper(function(result) {
        assertNull(result);
      }));
  testHelper.execute();
  callback.assertCalled();
}


function testExecuteScript_primitiveReturnValue() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.EXECUTE_SCRIPT).
          withParameters({
            'script': 'return document.body;',
            'args': []
          }).
          andReturnSuccess(123).
      replayAll();

  var driver = testHelper.createDriver();
  var callback;
  driver.executeScript('return document.body;').
      then(callback = callbackHelper(function(result) {
        assertEquals(123, result);
      }));
  testHelper.execute();
  callback.assertCalled();
}


function testExecuteScript_webElementReturnValue() {
  var json = {};
  json[webdriver.WebElement.ELEMENT_KEY] = 'foo';

  var testHelper = TestHelper.expectingSuccess().
      expect(CName.EXECUTE_SCRIPT).
          withParameters({
            'script': 'return document.body;',
            'args': []
          }).
          andReturnSuccess(json).
      replayAll();

  var driver = testHelper.createDriver();
  var callback;
  driver.executeScript('return document.body;').
      then(function(webelement) {
        webdriver.promise.when(webelement.id_,
            callback = callbackHelper(function(id) {
              webdriver.test.testutil.assertObjectEquals(id, json);
            }));
      });
  testHelper.execute();
  callback.assertCalled();
}


function testExecuteScript_arrayReturnValue() {
  var json = [{}];
  json[0][webdriver.WebElement.ELEMENT_KEY] = 'foo';

  var testHelper = TestHelper.expectingSuccess().
      expect(CName.EXECUTE_SCRIPT).
          withParameters({
            'script': 'return document.body;',
            'args': []
          }).
          andReturnSuccess(json).
      replayAll();

  var driver = testHelper.createDriver();
  var callback;
  driver.executeScript('return document.body;').
      then(function(array) {
        webdriver.promise.when(array[0].id_,
            callback = callbackHelper(function(id) {
              webdriver.test.testutil.assertObjectEquals(id, json[0]);
            }));
      });
  testHelper.execute();
  callback.assertCalled();
}


function testExecuteScript_objectReturnValue() {
  var json = {'foo':{}};
  json['foo'][webdriver.WebElement.ELEMENT_KEY] = 'foo';

  var testHelper = TestHelper.expectingSuccess().
      expect(CName.EXECUTE_SCRIPT).
          withParameters({
            'script': 'return document.body;',
            'args': []
          }).
          andReturnSuccess(json).
      replayAll();

  var driver = testHelper.createDriver();
  var callback;
  driver.executeScript('return document.body;').
      then(function(obj) {
        webdriver.promise.when(obj['foo'].id_,
            callback = callbackHelper(function(id) {
              webdriver.test.testutil.assertObjectEquals(id, json['foo']);
            }));
      });
  testHelper.execute();
  callback.assertCalled();
}


function testExecuteScript_scriptAsFunction() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.EXECUTE_SCRIPT).
          withParameters({
            'script': 'return (' + goog.nullFunction +
                      ').apply(null, arguments);',
            'args': []
          }).
          andReturnSuccess(null).
      replayAll();

  var driver = testHelper.createDriver();
  driver.executeScript(goog.nullFunction);
  testHelper.execute();
}


function testExecuteScript_simpleArgumentConversion() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.EXECUTE_SCRIPT).
          withParameters({
            'script': 'return 1;',
            'args': ['abc', 123, true, [123, {'foo': 'bar'}]]
          }).
          andReturnSuccess(null).
      replayAll();

  var driver = testHelper.createDriver();
  driver.executeScript('return 1;', 'abc', 123, true, [123, {'foo': 'bar'}]);
  testHelper.execute();
}


function testExecuteScript_webElementArgumentConversion() {
  var elementJson = {};
  elementJson[webdriver.WebElement.ELEMENT_KEY] = 'fefifofum';

  var testHelper = TestHelper.expectingSuccess().
      expect(CName.EXECUTE_SCRIPT).
          withParameters({
            'script': 'return 1;',
            'args': [elementJson]
          }).
          andReturnSuccess(null).
      replayAll();

  var driver = testHelper.createDriver();
  driver.executeScript('return 1;',
      new webdriver.WebElement(driver, elementJson));
  testHelper.execute();
}


function testExecuteScript_webElementPromiseArgumentConversion() {
  var elementJson = {'ELEMENT':'bar'};

  var testHelper = TestHelper.expectingSuccess().
      expect(CName.FIND_ELEMENT, {'using':'id', 'value':'foo'}).
          andReturnSuccess(elementJson).
      expect(CName.EXECUTE_SCRIPT).
          withParameters({
            'script': 'return 1;',
            'args': [elementJson]
          }).
          andReturnSuccess(null).
      replayAll();

  var driver = testHelper.createDriver();
  var element = driver.findElement(By.id('foo'));
  driver.executeScript('return 1;', element);
  testHelper.execute();
}


function testExecuteScript_argumentConversion() {
  var elementJson = {};
  elementJson[webdriver.WebElement.ELEMENT_KEY] = 'fefifofum';

  var testHelper = TestHelper.expectingSuccess().
      expect(CName.EXECUTE_SCRIPT).
          withParameters({
            'script': 'return 1;',
            'args': ['abc', 123, true, elementJson, [123, {'foo': 'bar'}]]
          }).
          andReturnSuccess(null).
      replayAll();

  var driver = testHelper.createDriver();
  var element = new webdriver.WebElement(driver, elementJson);
  driver.executeScript('return 1;',
      'abc', 123, true, element, [123, {'foo': 'bar'}]);
  testHelper.execute();
}


function testExecuteScript_scriptReturnsAnError() {
  var testHelper = TestHelper.
      expectingFailure(expectedError(ECode.UNKNOWN_ERROR, 'bam')).
      expect(CName.EXECUTE_SCRIPT).
          withParameters({
            'script': 'throw Error(arguments[0]);',
            'args': ['bam']
          }).
          andReturnError(ECode.UNKNOWN_ERROR, {'message':'bam'}).
      replayAll();
  var driver = testHelper.createDriver();
  driver.executeScript('throw Error(arguments[0]);', 'bam');
  testHelper.execute();
}


function testExecuteScript_failsIfArgumentIsARejectedPromise() {
  var testHelper = TestHelper.expectingSuccess().replayAll();

  var callback = callbackHelper(assertIsStubError);

  var arg = webdriver.promise.rejected(STUB_ERROR);
  arg.thenCatch(goog.nullFunction);  // Suppress default handler.

  var driver = testHelper.createDriver();
  driver.executeScript(goog.nullFunction, arg).thenCatch(callback);
  testHelper.execute();
  callback.assertCalled();
}


function testExecuteAsyncScript_failsIfArgumentIsARejectedPromise() {
  var testHelper = TestHelper.expectingSuccess().replayAll();

  var callback = callbackHelper(assertIsStubError);

  var arg = webdriver.promise.rejected(STUB_ERROR);
  arg.thenCatch(goog.nullFunction);  // Suppress default handler.

  var driver = testHelper.createDriver();
  driver.executeAsyncScript(goog.nullFunction, arg).thenCatch(callback);
  testHelper.execute();
  callback.assertCalled();
}


function testFindElement_elementNotFound() {
  var testHelper = TestHelper.
      expectingFailure(expectedError(ECode.NO_SUCH_ELEMENT, 'Unable to find element')).
      expect(CName.FIND_ELEMENT, {'using':'id', 'value':'foo'}).
      andReturnError(ECode.NO_SUCH_ELEMENT, {
          'message':'Unable to find element'
      }).
      replayAll();

  var driver = testHelper.createDriver();
  var element = driver.findElement(By.id('foo'));
  element.click();  // This should never execute.
  testHelper.execute();
}


function testFindElement_elementNotFoundInACallback() {
  var testHelper = TestHelper.
      expectingFailure(
          expectedError(ECode.NO_SUCH_ELEMENT, 'Unable to find element')).
      expect(CName.FIND_ELEMENT, {'using':'id', 'value':'foo'}).
      andReturnError(
          ECode.NO_SUCH_ELEMENT, {'message':'Unable to find element'}).
      replayAll();

  var driver = testHelper.createDriver();
  webdriver.promise.fulfilled().then(function() {
    var element = driver.findElement(By.id('foo'));
    return element.click();  // Should not execute.
  });
  testHelper.execute();
}


function testFindElement_elementFound() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.FIND_ELEMENT, {'using':'id', 'value':'foo'}).
          andReturnSuccess({'ELEMENT':'bar'}).
      expect(CName.CLICK_ELEMENT, {'id':{'ELEMENT':'bar'}}).
          andReturnSuccess().
      replayAll();

  var driver = testHelper.createDriver();
  var element = driver.findElement(By.id('foo'));
  element.click();
  testHelper.execute();
}


function testFindElement_canUseElementInCallback() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.FIND_ELEMENT, {'using':'id', 'value':'foo'}).
          andReturnSuccess({'ELEMENT':'bar'}).
      expect(CName.CLICK_ELEMENT, {'id':{'ELEMENT':'bar'}}).
          andReturnSuccess().
      replayAll();

  var driver = testHelper.createDriver();
  driver.findElement(By.id('foo')).then(function(element) {
    element.click();
  });
  testHelper.execute();
}


function testFindElement_byJs() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.EXECUTE_SCRIPT, {
        'script': 'return document.body',
        'args': []
      }).
      andReturnSuccess({'ELEMENT':'bar'}).
      expect(CName.CLICK_ELEMENT, {'id':{'ELEMENT':'bar'}}).
      replayAll();

  var driver = testHelper.createDriver();
  var element = driver.findElement(By.js('return document.body'));
  element.click();  // just to make sure
  testHelper.execute();
}


function testFindElement_byJs_returnsNonWebElementValue() {
  var testHelper = TestHelper.
      expectingFailure(function(e) {
        assertEquals(
            'Not the expected error message',
            'Custom locator did not return a WebElement', e.message);
      }).
      expect(CName.EXECUTE_SCRIPT, {'script': 'return 123', 'args': []}).
      andReturnSuccess(123).
      replayAll();

  var driver = testHelper.createDriver();
  var element = driver.findElement(By.js('return 123'));
  element.click();  // Should not execute.
  testHelper.execute();
}


function testFindElement_byJs_canPassArguments() {
  var script = 'return document.getElementsByTagName(arguments[0]);';
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.EXECUTE_SCRIPT, {
        'script': script,
        'args': ['div']
      }).
      andReturnSuccess({'ELEMENT':'one'}).
      replayAll();
  var driver = testHelper.createDriver();
  driver.findElement(By.js(script, 'div'));
  testHelper.execute();
}


function testFindElement_customLocator() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.FIND_ELEMENTS, {'using':'tag name', 'value':'a'}).
      andReturnSuccess([{'ELEMENT':'foo'}, {'ELEMENT':'bar'}]).
      expect(CName.CLICK_ELEMENT, {'id':{'ELEMENT':'foo'}}).
      andReturnSuccess().
      replayAll();

  var driver = testHelper.createDriver();
  var element = driver.findElement(function(d) {
    assertEquals(driver, d);
    return d.findElements(By.tagName('a'));
  });
  element.click();
  testHelper.execute();
}


function testFindElement_customLocatorThrowsIfResultIsNotAWebElement() {
  var testHelper = TestHelper.
      expectingFailure(function(e) {
        assertEquals(
            'Not the expected error message',
            'Custom locator did not return a WebElement', e.message);
      }).
      replayAll();

  var driver = testHelper.createDriver();
  driver.findElement(function() {
    return 1;
  });
  testHelper.execute();
}


function testIsElementPresent_elementNotFound() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.FIND_ELEMENTS, {'using':'id', 'value':'foo'}).
      andReturnSuccess([]).
      replayAll();

  var driver = testHelper.createDriver();
  var callback;
  driver.isElementPresent(By.id('foo')).
      then(callback = callbackHelper(function(result) {
        assertFalse(result);
      }));
  testHelper.execute();
  callback.assertCalled();
}


function testIsElementPresent_elementFound() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.FIND_ELEMENTS, {'using':'id', 'value':'foo'}).
      andReturnSuccess([{'ELEMENT':'bar'}]).
      replayAll();

  var driver = testHelper.createDriver();
  var callback;
  driver.isElementPresent(By.id('foo')).
      then(callback = callbackHelper(assertTrue));
  testHelper.execute();
  callback.assertCalled();
}


function testIsElementPresent_letsErrorsPropagate() {
  var testHelper = TestHelper.
      expectingFailure(expectedError(ECode.UNKNOWN_ERROR, 'There is no spoon')).
      expect(CName.FIND_ELEMENTS, {'using':'id', 'value':'foo'}).
      andReturnError(ECode.UNKNOWN_ERROR, {'message':'There is no spoon'}).
      replayAll();

  var driver = testHelper.createDriver();
  driver.isElementPresent(By.id('foo'));
  testHelper.execute();
}


function testIsElementPresent_byJs() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.EXECUTE_SCRIPT, {'script': 'return 123', 'args': []}).
      andReturnSuccess([{'ELEMENT':'bar'}]).
      replayAll();

  var driver = testHelper.createDriver();
  var callback;
  driver.isElementPresent(By.js('return 123')).
      then(callback = callbackHelper(function(result) {
        assertTrue(result);
      }));
  testHelper.execute();
  callback.assertCalled();
}


function testIsElementPresent_byJs_canPassScriptArguments() {
  var script = 'return document.getElementsByTagName(arguments[0]);';
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.EXECUTE_SCRIPT, {
        'script': script,
        'args': ['div']
      }).
      andReturnSuccess({'ELEMENT':'one'}).
      replayAll();

  var driver = testHelper.createDriver();
  driver.isElementPresent(By.js(script, 'div'));
  testHelper.execute();
}


function testFindElements() {
  var json = [
      {'ELEMENT':'foo'},
      {'ELEMENT':'bar'},
      {'ELEMENT':'baz'}
  ];
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.FIND_ELEMENTS, {'using':'tag name', 'value':'a'}).
      andReturnSuccess(json).
      replayAll();

  var driver = testHelper.createDriver();
  var callbacks = [];
  driver.findElements(By.tagName('a')).then(function(elements) {
    assertEquals(3, elements.length);

    function assertTypeAndId(index) {
      assertTrue('Not a WebElement at index ' + index,
          elements[index] instanceof webdriver.WebElement);
      elements[index].getId().
          then(callbacks[index] = callbackHelper(function(id) {
            webdriver.test.testutil.assertObjectEquals(json[index], id);
          }));
    }

    assertTypeAndId(0);
    assertTypeAndId(1);
    assertTypeAndId(2);
  });

  testHelper.execute();
  assertEquals(3, callbacks.length);
  callbacks[0].assertCalled();
  callbacks[1].assertCalled();
  callbacks[2].assertCalled();
}


function testFindElements_byJs() {
  var json = [
      {'ELEMENT':'foo'},
      {'ELEMENT':'bar'},
      {'ELEMENT':'baz'}
  ];
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.EXECUTE_SCRIPT, {
        'script': 'return document.getElementsByTagName("div");',
        'args': []
      }).
      andReturnSuccess(json).
      replayAll();

  var driver = testHelper.createDriver();
  var callbacks = [];
  driver.findElements(By.js('return document.getElementsByTagName("div");')).
      then(function(elements) {
        assertEquals(3, elements.length);

        function assertTypeAndId(index) {
          assertTrue('Not a WebElement at index ' + index,
              elements[index] instanceof webdriver.WebElement);
          elements[index].getId().
              then(callbacks[index] = callbackHelper(function(id) {
                webdriver.test.testutil.assertObjectEquals(json[index], id);
              }));
        }

        assertTypeAndId(0);
        assertTypeAndId(1);
        assertTypeAndId(2);
      });

  testHelper.execute();
  assertEquals(3, callbacks.length);
  callbacks[0].assertCalled();
  callbacks[1].assertCalled();
  callbacks[2].assertCalled();
}


function testFindElements_byJs_filtersOutNonWebElementResponses() {
  var json = [
      {'ELEMENT':'foo'},
      123,
      'a',
      false,
      {'ELEMENT':'bar'},
      {'not a web element': 1},
      {'ELEMENT':'baz'}
  ];
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.EXECUTE_SCRIPT, {
        'script': 'return document.getElementsByTagName("div");',
        'args': []
      }).
      andReturnSuccess(json).
      replayAll();

  var driver = testHelper.createDriver();
  var callbacks = [];
  driver.findElements(By.js('return document.getElementsByTagName("div");')).
      then(function(elements) {
        assertEquals(3, elements.length);

        function assertTypeAndId(index, jsonIndex) {
          assertTrue('Not a WebElement at index ' + index,
              elements[index] instanceof webdriver.WebElement);
          elements[index].getId().
              then(callbacks[index] = callbackHelper(function(id) {
                webdriver.test.testutil.assertObjectEquals(json[jsonIndex], id);
              }));
        }

        assertTypeAndId(0, 0);
        assertTypeAndId(1, 4);
        assertTypeAndId(2, 6);
      });

  testHelper.execute();
  assertEquals(3, callbacks.length);
  callbacks[0].assertCalled();
  callbacks[1].assertCalled();
  callbacks[2].assertCalled();
}


function testFindElements_byJs_convertsSingleWebElementResponseToArray() {
  var json = {'ELEMENT':'foo'};
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.EXECUTE_SCRIPT, {
        'script': 'return document.getElementsByTagName("div");',
        'args': []
      }).
      andReturnSuccess(json).
      replayAll();

  var driver = testHelper.createDriver();
  var callback1, callback2;
  driver.findElements(By.js('return document.getElementsByTagName("div");')).
      then(callback1 = callbackHelper(function(elements) {
        assertEquals(1, elements.length);
        assertTrue(elements[0] instanceof webdriver.WebElement);
        elements[0].getId().
            then(callback2 = callbackHelper(function(id) {
              webdriver.test.testutil.assertObjectEquals(json, id);
            }));
      }));

  testHelper.execute();
  callback1.assertCalled();
  callback2.assertCalled();
}


function testFindElements_byJs_canPassScriptArguments() {
  var script = 'return document.getElementsByTagName(arguments[0]);';
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.EXECUTE_SCRIPT, {
        'script': script,
        'args': ['div']
      }).
      andReturnSuccess([{'ELEMENT':'one'}, {'ELEMENT':'two'}]).
      replayAll();

  var driver = testHelper.createDriver();
  driver.findElements(By.js(script, 'div'));
  testHelper.execute();
}


function testSendKeysConvertsVarArgsIntoStrings_simpleArgs() {
  var testHelper = TestHelper.
      expectingSuccess().
      expect(CName.SEND_KEYS_TO_ELEMENT, {'id':{'ELEMENT':'one'},
                                          'value':['1','2','abc','3']}).
          andReturnSuccess().
      replayAll();

  var driver = testHelper.createDriver();
  var element = new webdriver.WebElement(driver, {'ELEMENT': 'one'});
  element.sendKeys(1, 2, 'abc', 3);
  testHelper.execute();
}


function testSendKeysConvertsVarArgsIntoStrings_promisedArgs() {
  var testHelper = TestHelper.
      expectingSuccess().
      expect(CName.FIND_ELEMENT, {'using':'id', 'value':'foo'}).
          andReturnSuccess({'ELEMENT':'one'}).
      expect(CName.SEND_KEYS_TO_ELEMENT, {'id':{'ELEMENT':'one'},
                                          'value':['abc', '123', 'def']}).
          andReturnSuccess().
      replayAll();

  var driver = testHelper.createDriver();
  var element = driver.findElement(By.id('foo'));
  element.sendKeys(
      webdriver.promise.fulfilled('abc'), 123,
      webdriver.promise.fulfilled('def'));
  testHelper.execute();
}

function testElementEquality_isReflexive() {
  var a = new webdriver.WebElement(STUB_DRIVER, 'foo');
  var callback;
  webdriver.WebElement.equals(a, a).then(
      callback = callbackHelper(assertTrue));
  callback.assertCalled();
  verifyAll();  // for tearDown()
}

function testElementEquals_doesNotSendRpcIfElementsHaveSameId() {
  var a = new webdriver.WebElement(STUB_DRIVER, 'foo'),
      b = new webdriver.WebElement(STUB_DRIVER, 'foo'),
      c = new webdriver.WebElement(STUB_DRIVER, 'foo'),
      callback;

  webdriver.WebElement.equals(a, b).then(
      callback = callbackHelper(assertTrue));
  callback.assertCalled('a should == b!');
  webdriver.WebElement.equals(b, a).then(
      callback = callbackHelper(assertTrue));
  callback.assertCalled('symmetry check failed');
  webdriver.WebElement.equals(a, c).then(
      callback = callbackHelper(assertTrue));
  callback.assertCalled('a should == c!');
  webdriver.WebElement.equals(b, c).then(
      callback = callbackHelper(assertTrue));
  callback.assertCalled('transitive check failed');

  verifyAll();  // for tearDown()
}

function testElementEquals_sendsRpcIfElementsHaveDifferentIds() {
  var id1 = {'ELEMENT':'foo'};
  var id2 = {'ELEMENT':'bar'};
  var testHelper = TestHelper.
      expectingSuccess().
      expect(CName.ELEMENT_EQUALS, {'id':id1, 'other':id2}).
      andReturnSuccess(true).
      replayAll();

  var driver = testHelper.createDriver();
  var a = new webdriver.WebElement(driver, id1),
      b = new webdriver.WebElement(driver, id2),
      callback;

  webdriver.WebElement.equals(a, b).then(
      callback = callbackHelper(assertTrue));

  testHelper.execute();
  callback.assertCalled();
}


function testElementEquals_failsIfAnInputElementCouldNotBeFound() {
  var testHelper = TestHelper.expectingSuccess().replayAll();

  var callback = callbackHelper(assertIsStubError);
  var id = webdriver.promise.rejected(STUB_ERROR);
  id.thenCatch(goog.nullFunction);  // Suppress default handler.

  var driver = testHelper.createDriver();
  var a = new webdriver.WebElement(driver, {'ELEMENT': 'foo'});
  var b = new webdriver.WebElementPromise(driver, id);

  webdriver.WebElement.equals(a, b).thenCatch(callback);
  testHelper.execute();
  callback.assertCalled();
}

function testWaiting_waitSucceeds() {
  var testHelper = TestHelper.expectingSuccess().
      expect(CName.FIND_ELEMENTS, {'using':'id', 'value':'foo'}).
      andReturnSuccess([]).
      expect(CName.FIND_ELEMENTS, {'using':'id', 'value':'foo'}).
      andReturnSuccess([]).
      expect(CName.FIND_ELEMENTS, {'using':'id', 'value':'foo'}).
      andReturnSuccess([{'ELEMENT':'bar'}]).
      replayAll();

  var driver = testHelper.createDriver();
  driver.wait(function() {
    return driver.isElementPresent(By.id('foo'));
  }, 200);
  testHelper.execute();
}


function testWaiting_waitTimesout() {
  var testHelper = TestHelper.
      expectingFailure(function(e) {
        assertEquals('Wait timed out after ',
            e.message.substring(0, 'Wait timed out after '.length));
      }).
      expect(CName.FIND_ELEMENTS, {'using':'id', 'value':'foo'}).
      andReturnSuccess([]).
      expect(CName.FIND_ELEMENTS, {'using':'id', 'value':'foo'}).
      andReturnSuccess([]).
      expect(CName.FIND_ELEMENTS, {'using':'id', 'value':'foo'}).
      andReturnSuccess([]).
      replayAll();

  var driver = testHelper.createDriver();
  driver.wait(function() {
    return driver.isElementPresent(By.id('foo'));
  }, 200);
  testHelper.execute();
}

function testInterceptsAndTransformsUnhandledAlertErrors() {
  var testHelper = TestHelper.
      expectingSuccess().
      expect(CName.FIND_ELEMENT, {'using':'id', 'value':'foo'}).
      andReturnError(ECode.UNEXPECTED_ALERT_OPEN, {
        'message': 'boom',
        'alert': {'text': 'hello'}
      }).
      replayAll();

  var pair = callbackPair(null, function(e) {
    assertTrue(e instanceof webdriver.UnhandledAlertError);

    var pair = callbackPair(goog.partial(assertEquals, 'hello'));
    e.getAlert().getText().then(pair.callback, pair.errback);
    pair.assertCallback();
  });

  var driver = testHelper.createDriver();
  driver.findElement(By.id('foo')).then(pair.callback, pair.errback);
  testHelper.execute();
  pair.assertErrback();
}

function testUnhandledAlertErrors_usesEmptyStringIfAlertTextOmittedFromResponse() {
  var testHelper = TestHelper.
      expectingSuccess().
      expect(CName.FIND_ELEMENT, {'using':'id', 'value':'foo'}).
      andReturnError(ECode.UNEXPECTED_ALERT_OPEN, {'message': 'boom'}).
      replayAll();

  var pair = callbackPair(null, function(e) {
    assertTrue(e instanceof webdriver.UnhandledAlertError);

    var pair = callbackPair(goog.partial(assertEquals, ''));
    e.getAlert().getText().then(pair.callback, pair.errback);
    pair.assertCallback();
  });

  var driver = testHelper.createDriver();
  driver.findElement(By.id('foo')).then(pair.callback, pair.errback);
  testHelper.execute();
  pair.assertErrback();
}

function testAlertHandleResolvesWhenPromisedTextResolves() {
  var promise = new webdriver.promise.Deferred();

  var alert = new webdriver.AlertPromise(STUB_DRIVER, promise);
  assertTrue(alert.isPending());

  var callback;
  webdriver.promise.when(alert.getText(),
      callback = callbackHelper(function(text) {
        assertEquals('foo', text);
      }));

  callback.assertNotCalled();

  promise.fulfill(new webdriver.Alert(STUB_DRIVER, 'foo'));

  callback.assertCalled();
  verifyAll();  // Make tearDown happy.
}


function testWebElementsBelongToSameFlowAsParentDriver() {
  var testHelper = TestHelper
      .expectingSuccess()
      .expect(CName.FIND_ELEMENT, {'using':'id', 'value':'foo'})
      .andReturnSuccess({'ELEMENT': 'abc123'})
      .replayAll();

  var driver = testHelper.createDriver();
  webdriver.promise.createFlow(function() {
    driver.findElement({id: 'foo'}).then(function() {
      assertEquals(
          'WebElement should belong to the same flow as its parent driver',
          driver.controlFlow(), webdriver.promise.controlFlow());
    });
  });

  testHelper.execute();
}


function testSwitchToAlertThatIsNotPresent() {
  var testHelper = TestHelper
      .expectingFailure(expectedError(ECode.NO_SUCH_ALERT, 'no alert'))
      .expect(CName.GET_ALERT_TEXT)
      .andReturnError(ECode.NO_SUCH_ALERT, {'message': 'no alert'})
      .replayAll();

  var driver = testHelper.createDriver();
  var alert = driver.switchTo().alert();
  alert.dismiss();  // Should never execute.
  testHelper.execute();
}


function testAlertsBelongToSameFlowAsParentDriver() {
  var testHelper = TestHelper
      .expectingSuccess()
      .expect(CName.GET_ALERT_TEXT).andReturnSuccess('hello')
      .replayAll();

  var driver = testHelper.createDriver();
  webdriver.promise.createFlow(function() {
    driver.switchTo().alert().then(function() {
      assertEquals(
          'Alert should belong to the same flow as its parent driver',
          driver.controlFlow(), webdriver.promise.controlFlow());
    });
  });

  testHelper.execute();
}

function testFetchingLogs() {
  var testHelper = TestHelper.
      expectingSuccess().
      expect(CName.GET_LOG, {'type': 'browser'}).
      andReturnSuccess([
        new webdriver.logging.Entry(
            webdriver.logging.Level.INFO, 'hello', 1234),
        {'level': 'DEBUG', 'message': 'abc123', 'timestamp': 5678}
      ]).
      replayAll();

  var pair = callbackPair(function(entries) {
    assertEquals(2, entries.length);

    assertTrue(entries[0] instanceof webdriver.logging.Entry);
    assertEquals(webdriver.logging.Level.INFO.value, entries[0].level.value);
    assertEquals('hello', entries[0].message);
    assertEquals(1234, entries[0].timestamp);

    assertTrue(entries[1] instanceof webdriver.logging.Entry);
    assertEquals(webdriver.logging.Level.DEBUG.value, entries[1].level.value);
    assertEquals('abc123', entries[1].message);
    assertEquals(5678, entries[1].timestamp);
  });

  var driver = testHelper.createDriver();
  driver.manage().logs().get('browser').then(pair.callback, pair.errback);
  testHelper.execute();
  pair.assertCallback();
}


function testCommandsFailIfInitialSessionCreationFailed() {
  var testHelper = TestHelper.expectingSuccess().replayAll();
  var navigateResult = callbackPair(null, assertIsStubError);
  var quitResult = callbackPair(null, assertIsStubError);

  var session = webdriver.promise.rejected(STUB_ERROR);

  var driver = testHelper.createDriver(session);
  driver.get('some-url').then(navigateResult.callback, navigateResult.errback);
  driver.quit().then(quitResult.callback, quitResult.errback);

  testHelper.execute();
  navigateResult.assertErrback();
  quitResult.assertErrback();
}


function testWebElementCommandsFailIfInitialDriverCreationFailed() {
  var testHelper = TestHelper.expectingSuccess().replayAll();

  var session = webdriver.promise.rejected(STUB_ERROR);
  var callback = callbackHelper(assertIsStubError);

  var driver = testHelper.createDriver(session);
  driver.findElement(By.id('foo')).click().thenCatch(callback);
  testHelper.execute();
  callback.assertCalled();
}


function testWebElementCommansFailIfElementCouldNotBeFound() {
  var testHelper = TestHelper.
      expectingSuccess().
      expect(CName.FIND_ELEMENT, {'using':'id', 'value':'foo'}).
          andReturnError(ECode.NO_SUCH_ELEMENT,
                         {'message':'Unable to find element'}).
      replayAll();

  var callback = callbackHelper(
      expectedError(ECode.NO_SUCH_ELEMENT, 'Unable to find element'));

  var driver = testHelper.createDriver();
  driver.findElement(By.id('foo')).click().thenCatch(callback);
  testHelper.execute();
  callback.assertCalled();
}


function testCannotFindChildElementsIfParentCouldNotBeFound() {
  var testHelper = TestHelper.
      expectingSuccess().
      expect(CName.FIND_ELEMENT, {'using':'id', 'value':'foo'}).
      andReturnError(ECode.NO_SUCH_ELEMENT,
                     {'message':'Unable to find element'}).
      replayAll();

  var callback = callbackHelper(
      expectedError(ECode.NO_SUCH_ELEMENT, 'Unable to find element'));

  var driver = testHelper.createDriver();
  driver.findElement(By.id('foo'))
      .findElement(By.id('bar'))
      .findElement(By.id('baz'))
      .thenCatch(callback);
  testHelper.execute();
  callback.assertCalled();
}


function testActionSequenceFailsIfInitialDriverCreationFailed() {
  var testHelper = TestHelper.expectingSuccess().replayAll();

  var session = webdriver.promise.rejected(STUB_ERROR);

  // Suppress the default error handler so we can verify it propagates
  // to the perform() call below.
  session.thenCatch(goog.nullFunction);

  var callback = callbackHelper(assertIsStubError);

  var driver = testHelper.createDriver(session);
  driver.actions().
      mouseDown().
      mouseUp().
      perform().
      thenCatch(callback);
  testHelper.execute();
  callback.assertCalled();
}


function testAlertCommandsFailIfAlertNotPresent() {
  var testHelper = TestHelper
      .expectingSuccess()
      .expect(CName.GET_ALERT_TEXT)
      .andReturnError(ECode.NO_SUCH_ALERT, {'message': 'no alert'})
      .replayAll();

  var driver = testHelper.createDriver();
  var alert = driver.switchTo().alert();

  var expectError = expectedError(ECode.NO_SUCH_ALERT, 'no alert');
  var callbacks = [];
  for (var key in webdriver.Alert.prototype) {
    if (webdriver.Alert.prototype.hasOwnProperty(key)) {
      var helper = callbackHelper(expectError);
      callbacks.push(key, helper);
      alert[key].call(alert).thenCatch(helper);
    }
  }

  testHelper.execute();
  for (var i = 0; i < callbacks.length - 1; i += 2) {
    callbacks[i + 1].assertCalled(
            'Error did not propagate for ' + callbacks[i]);
  }
}
