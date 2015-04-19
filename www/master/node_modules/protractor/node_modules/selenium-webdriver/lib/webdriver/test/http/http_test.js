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
goog.require('goog.Uri');
goog.require('goog.json');
goog.require('goog.testing.MockControl');
goog.require('goog.testing.jsunit');
goog.require('webdriver.Command');
goog.require('webdriver.http.Client');
goog.require('webdriver.http.Executor');
goog.require('webdriver.promise');
goog.require('webdriver.test.testutil');

// Alias for readability.
var callbackHelper = webdriver.test.testutil.callbackHelper;

var control = new goog.testing.MockControl();
var mockClient, executor, onCallback, onErrback;

function setUp() {
  mockClient = control.createStrictMock(webdriver.http.Client);

  executor = new webdriver.http.Executor(mockClient);
}

function tearDown() {
  control.$tearDown();
}

function assertSuccess() {
  onErrback.assertNotCalled('Did not expect errback');
  onCallback.assertCalled('Expected callback');
}

function assertFailure() {
  onCallback.assertNotCalled('Did not expect callback');
  onErrback.assertCalled('Expected errback');
}

function headersToString(headers) {
  var str = [];
  for (var key in headers) {
    str.push(key + ': ' + headers[key]);
  }
  return str.join('\n');
}

function expectRequest(method, path, data, headers) {
  var description = method + ' ' + path + '\n' + headersToString(headers) +
                    '\n' + goog.json.serialize(data);

  return mockClient.send(new goog.testing.mockmatchers.ArgumentMatcher(
      function(request) {
        assertEquals('wrong method', method, request.method);
        assertEquals('wrong path', path + '', request.path);
        webdriver.test.testutil.assertObjectEquals(data, request.data);
        assertNull(
            'Wrong headers for request:\n' + description +
            '\n    Actual headers were:\n' + headersToString(request.headers),
            goog.testing.asserts.findDifferences(headers, request.headers));
        return true;
      }, description),
      goog.testing.mockmatchers.isFunction);
}

function response(status, headers, body) {
  return new webdriver.http.Response(status, headers, body);
}

function respondsWith(error, opt_response) {
  return function(request, callback) {
    callback(error, opt_response);
  };
}

///////////////////////////////////////////////////////////////////////////////
//
//  Tests
//
///////////////////////////////////////////////////////////////////////////////

function testBuildPath() {
  var parameters = {'sessionId':'foo', 'url':'http://www.google.com'};
  var finalPath = webdriver.http.Executor.buildPath_(
      '/session/:sessionId/url', parameters);
  assertEquals('/session/foo/url', finalPath);
  webdriver.test.testutil.assertObjectEquals({'url':'http://www.google.com'},
      parameters);
}

function testBuildPath_withWebElement() {
  var parameters = {'sessionId':'foo', 'id': {}};
  parameters['id']['ELEMENT'] = 'bar';

  var finalPath = webdriver.http.Executor.buildPath_(
      '/session/:sessionId/element/:id/click', parameters);
  assertEquals('/session/foo/element/bar/click', finalPath);
  webdriver.test.testutil.assertObjectEquals({}, parameters);
}

function testBuildPath_throwsIfMissingParameter() {
  assertThrows(goog.partial(webdriver.http.Executor.buildPath_,
      '/session/:sessionId', {}));

  assertThrows(goog.partial(webdriver.http.Executor.buildPath_,
      '/session/:sessionId/element/:id', {'sessionId': 'foo'}));
}

function testBuildPath_doesNotMatchOnSegmentsThatDoNotStartWithColon() {
  assertEquals('/session/foo:bar/baz',
      webdriver.http.Executor.buildPath_('/session/foo:bar/baz', {}));
}

function testExecute_rejectsUnrecognisedCommands() {
  assertThrows(goog.bind(executor.execute, executor,
      new webdriver.Command('fake-command-name'), goog.nullFunction));
}

/**
 * @param {!webdriver.Command} command The command to send.
 * @param {!Function=} opt_onSuccess The function to check the response with.
 */
function assertSendsSuccessfully(command, opt_onSuccess) {
  var callback;
  executor.execute(command, callback = callbackHelper(function(e, response) {
    assertNull(e);
    assertNotNullNorUndefined(response);
    if (opt_onSuccess) {
      opt_onSuccess(response);
    }
  }));
  callback.assertCalled();
  control.$verifyAll();
}

/**
 * @param {!webdriver.Command} command The command to send.
 * @param {!Function=} opt_onError The function to check the error with.
 */
function assertFailsToSend(command, opt_onError) {
  var callback;
  executor.execute(command, callback = callbackHelper(function(e, response) {
    assertNotNullNorUndefined(e);
    assertUndefined(response);
    if (opt_onError) {
      opt_onError(e);
    }
  }));
  callback.assertCalled();
  control.$verifyAll();
}

function testExecute_clientFailsToSendRequest() {
  var error = new Error('boom');
  expectRequest('POST', '/session', {}, {
    'Accept': 'application/json; charset=utf-8'
  }).
  $does(respondsWith(error));
  control.$replayAll();

  assertFailsToSend(new webdriver.Command(webdriver.CommandName.NEW_SESSION),
      function(e) {
        assertEquals(error, e);
      });
}

function testExecute_commandWithNoUrlParameters() {
  expectRequest('POST', '/session', {}, {
    'Accept': 'application/json; charset=utf-8'
  }).
  $does(respondsWith(null, response(200, {}, '')));
  control.$replayAll();

  assertSendsSuccessfully(
      new webdriver.Command(webdriver.CommandName.NEW_SESSION));
}

function testExecute_rejectsCommandsMissingUrlParameters() {
  var command =
      new webdriver.Command(webdriver.CommandName.FIND_CHILD_ELEMENT).
          setParameter('sessionId', 's123').
          // Let this be missing: setParameter('id', {'ELEMENT': 'e456'}).
          setParameter('using', 'id').
          setParameter('value', 'foo');

  control.$replayAll();
  assertThrows(goog.bind(executor.execute, executor, command));
  control.$verifyAll();
}

function testExecute_replacesUrlParametersWithCommandParameters() {
  var command =
      new webdriver.Command(webdriver.CommandName.GET).
          setParameter('sessionId', 's123').
          setParameter('url', 'http://www.google.com');

  expectRequest('POST', '/session/s123/url',
      {'url': 'http://www.google.com'},
      {'Accept': 'application/json; charset=utf-8'}).
      $does(respondsWith(null, response(200, {}, '')));
  control.$replayAll();

  assertSendsSuccessfully(command);
}

function testExecute_returnsParsedJsonResponse() {
  var responseObj = {
    'status': bot.ErrorCode.SUCCESS,
    'value': 'http://www.google.com'
  };
  var command = new webdriver.Command(webdriver.CommandName.GET_CURRENT_URL).
      setParameter('sessionId', 's123');

  expectRequest('GET', '/session/s123/url', {}, {
    'Accept': 'application/json; charset=utf-8'
  }).$does(respondsWith(null,
      response(200, {'Content-Type': 'application/json'},
          goog.json.serialize(responseObj))));
  control.$replayAll();

  assertSendsSuccessfully(command, function(response) {
    webdriver.test.testutil.assertObjectEquals(responseObj, response);
  });
}

function testExecute_returnsSuccessFor2xxWithBodyAsValueWhenNotJson() {
  var command = new webdriver.Command(webdriver.CommandName.GET_CURRENT_URL).
      setParameter('sessionId', 's123');

  expectRequest('GET', '/session/s123/url', {}, {
    'Accept': 'application/json; charset=utf-8'
  }).$does(respondsWith(null,
      response(200, {}, 'hello, world\r\ngoodbye, world!')));
  control.$replayAll();

  assertSendsSuccessfully(command, function(response) {
    webdriver.test.testutil.assertObjectEquals({
      'status': bot.ErrorCode.SUCCESS,
      'value': 'hello, world\ngoodbye, world!'
    }, response);
  });
}

function testExecute_returnsSuccessFor2xxInvalidJsonBody() {
  var invalidJson = '[';
  expectRequest('POST', '/session', {}, {
    'Accept': 'application/json; charset=utf-8'
  }).
  $does(respondsWith(null, response(200, {
    'Content-Type': 'application/json'
  }, invalidJson)));
  control.$replayAll();

  assertSendsSuccessfully(
      new webdriver.Command(webdriver.CommandName.NEW_SESSION),
      function(response) {
        webdriver.test.testutil.assertObjectEquals({
          'status': bot.ErrorCode.SUCCESS,
          'value': invalidJson
        }, response);
      });
}

function testExecute_returnsUnknownCommandFor404WithBodyAsValueWhenNotJson() {
  var command = new webdriver.Command(webdriver.CommandName.GET_CURRENT_URL).
      setParameter('sessionId', 's123');

  expectRequest('GET', '/session/s123/url', {}, {
    'Accept': 'application/json; charset=utf-8'
  }).$does(respondsWith(null,
      response(404, {}, 'hello, world\r\ngoodbye, world!')));
  control.$replayAll();

  assertSendsSuccessfully(command, function(response) {
    webdriver.test.testutil.assertObjectEquals({
      'status': bot.ErrorCode.UNKNOWN_COMMAND,
      'value': 'hello, world\ngoodbye, world!'
    }, response);
  });
}

function testExecute_returnsUnknownErrorForGenericErrorCodeWithBodyAsValueWhenNotJson() {
  var command = new webdriver.Command(webdriver.CommandName.GET_CURRENT_URL).
      setParameter('sessionId', 's123');

  expectRequest('GET', '/session/s123/url', {}, {
    'Accept': 'application/json; charset=utf-8'
  }).$does(respondsWith(null,
      response(500, {}, 'hello, world\r\ngoodbye, world!')));
  control.$replayAll();

  assertSendsSuccessfully(command, function(response) {
    webdriver.test.testutil.assertObjectEquals({
      'status': bot.ErrorCode.UNKNOWN_ERROR,
      'value': 'hello, world\ngoodbye, world!'
    }, response);
  });
}

function testExecute_attemptsToParseBodyWhenNoContentTypeSpecified() {
  var responseObj = {
    'status': bot.ErrorCode.SUCCESS,
    'value': 'http://www.google.com'
  };
  var command = new webdriver.Command(webdriver.CommandName.GET_CURRENT_URL).
      setParameter('sessionId', 's123');

  expectRequest('GET', '/session/s123/url', {}, {
    'Accept': 'application/json; charset=utf-8'
  }).$does(respondsWith(null,
      response(200, {}, goog.json.serialize(responseObj))));
  control.$replayAll();

  assertSendsSuccessfully(command, function(response) {
    webdriver.test.testutil.assertObjectEquals(responseObj, response);
  });
}

function FakeXmlHttpRequest(headers, status, responseText) {
  return {
    getAllResponseHeaders: function() { return headers; },
    status: status,
    responseText: responseText
  };
}

function testXmlHttpRequestToHttpResponse_parseHeaders_windows() {
  var response = webdriver.http.Response.fromXmlHttpRequest(
      FakeXmlHttpRequest([
        'a:b',
        'c: d',
        'e :f',
        'g : h'
      ].join('\r\n'), 200, ''));
  assertEquals(200, response.status);
  assertEquals('', response.body);

  webdriver.test.testutil.assertObjectEquals({
    'a': 'b',
    'c': 'd',
    'e': 'f',
    'g': 'h'
  }, response.headers);
}

function testXmlHttpRequestToHttpResponse_parseHeaders_unix() {
  var response = webdriver.http.Response.fromXmlHttpRequest(
      FakeXmlHttpRequest([
        'a:b',
        'c: d',
        'e :f',
        'g : h'
      ].join('\n'), 200, ''));
  assertEquals(200, response.status);
  assertEquals('', response.body);

  webdriver.test.testutil.assertObjectEquals({
    'a': 'b',
    'c': 'd',
    'e': 'f',
    'g': 'h'
  }, response.headers);
}

function testXmlHttpRequestToHttpResponse_noHeaders() {
  var response = webdriver.http.Response.fromXmlHttpRequest(
      FakeXmlHttpRequest('', 200, ''));
  assertEquals(200, response.status);
  assertEquals('', response.body);
  webdriver.test.testutil.assertObjectEquals({}, response.headers);
}

function testXmlHttpRequestToHttpResponse_stripsNullCharactersFromBody() {
  var response = webdriver.http.Response.fromXmlHttpRequest(
      FakeXmlHttpRequest('', 200, '\x00\0foo\x00\x00bar\x00\0'));
  assertEquals(200, response.status);
  assertEquals('foobar', response.body);
  webdriver.test.testutil.assertObjectEquals({}, response.headers);
}
