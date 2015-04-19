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
goog.require('goog.string');
goog.require('goog.testing.JsUnitException');
goog.require('goog.testing.PropertyReplacer');
goog.require('goog.testing.StrictMock');
goog.require('goog.testing.jsunit');
goog.require('goog.testing.stacktrace');
goog.require('webdriver.stacktrace');
goog.require('webdriver.test.testutil');

var stubs;

function setUpPage() {
  stubs = new goog.testing.PropertyReplacer();
}

function tearDown() {
  stubs.reset();
}


function assertStackFrame(message, frameOrFrameString, expectedFrame) {
  var frame = frameOrFrameString;
  if (goog.isString(frame)) {
    frame = webdriver.stacktrace.parseStackFrame_(frame);
    assertNotNull(message + '\nunable to parse frame: ' + frameOrFrameString,
        frame);
  }
  var diff = [];
  for (var prop in expectedFrame) {
    if (goog.isString(expectedFrame[prop]) &&
        frame[prop] !== expectedFrame[prop]) {
      diff.push([
        prop, ': <', expectedFrame[prop], '> !== <', frame[prop], '>'
      ].join(''));
    }
  }
  if (diff.length) {
    fail(message +
        '\nfor:      <' + frameOrFrameString + '>' +
        '\nexpected: <' + expectedFrame + '>' +
        '\nbut was:  <' + frame + '>' +
        '\n    ' +
        diff.join('\n    '));
  }
}

function assertFrame(frame, file, line) {
  assertContains('/' + file, frame.getUrl());
  assertEquals(line, frame.getLine());
}

function testGetStacktraceFromFile() {
  if (!webdriver.stacktrace.BROWSER_SUPPORTED) {
    return false;
  }

  var stacktrace = webdriver.test.testutil.getStackTrace();
  assertFrame(stacktrace[0], 'testutil.js', 36);
  assertFrame(stacktrace[1], 'stacktrace_test.js', 73);
}

function testGetStacktraceWithUrlOnLine() {
  if (!webdriver.stacktrace.BROWSER_SUPPORTED) {
    return false;
  }

  // The url argument is intentionally ignored here.
  function getStacktraceWithUrlArgument(url) {
    return webdriver.stacktrace.get();
  }

  var stacktrace = getStacktraceWithUrlArgument('http://www.google.com');
  assertFrame(stacktrace[0], 'stacktrace_test.js', 85);

  stacktrace = getStacktraceWithUrlArgument('http://www.google.com/search');
  assertFrame(stacktrace[0], 'stacktrace_test.js', 85);
}

function testParseStackFrameInV8() {
  assertStackFrame('exception name only (node v0.8)',
      '    at Error (unknown source)',
      new webdriver.stacktrace.Frame('', 'Error', '', 'unknown source'));

  assertStackFrame('exception name only (chrome v22)',
      '    at Error (<anonymous>)',
      new webdriver.stacktrace.Frame('', 'Error', '', '<anonymous>'));

  assertStackFrame('context object + function name + url',
      '    at Object.assert (file:///.../asserts.js:29:10)',
      new webdriver.stacktrace.Frame('Object', 'assert', '',
          'file:///.../asserts.js:29:10'));

  assertStackFrame('context object + function name + file',
      '    at Object.assert (asserts.js:29:10)',
      new webdriver.stacktrace.Frame('Object', 'assert', '',
          'asserts.js:29:10'));

  assertStackFrame('context object + anonymous function + file',
      '    at Interface.<anonymous> (repl.js:182:12)',
      new webdriver.stacktrace.Frame('Interface', '<anonymous>', '',
          'repl.js:182:12'));

  assertStackFrame('url only',
      '    at http://www.example.com/jsunit.js:117:13',
      new webdriver.stacktrace.Frame('', '', '',
          'http://www.example.com/jsunit.js:117:13'));

  assertStackFrame('file only',
      '    at repl:1:57',
      new webdriver.stacktrace.Frame('', '', '', 'repl:1:57'));

  assertStackFrame('function alias',
      '    at [object Object].exec [as execute] (file:///foo)',
      new webdriver.stacktrace.Frame('[object Object]', 'exec',
          'execute', 'file:///foo'));

  assertStackFrame('constructor call',
      '    at new Class (file:///foo)',
      new webdriver.stacktrace.Frame('new ', 'Class', '', 'file:///foo'));

  assertStackFrame('object property constructor call',
      '    at new [object Object].foo (repl:1:2)',
      new webdriver.stacktrace.Frame('new [object Object]', 'foo', '',
          'repl:1:2'));

  assertStackFrame('namespaced constructor call',
      '    at new foo.bar.Class (foo:1:2)',
      new webdriver.stacktrace.Frame('new foo.bar', 'Class', '', 'foo:1:2'));

  assertStackFrame('anonymous constructor call',
      '    at new <anonymous> (file:///foo)',
      new webdriver.stacktrace.Frame('new ', '<anonymous>', '',
          'file:///foo'));

  assertStackFrame('native function call',
      '    at Array.forEach (native)',
      new webdriver.stacktrace.Frame('Array', 'forEach', '', 'native'));

  assertStackFrame('eval',
      '    at foo (eval at file://bar)',
      new webdriver.stacktrace.Frame('', 'foo', '', 'eval at file://bar'));

  assertStackFrame('nested anonymous eval',
      '    at eval (eval at <anonymous> (unknown source), <anonymous>:2:7)',
      new webdriver.stacktrace.Frame('', 'eval', '',
          'eval at <anonymous> (unknown source), <anonymous>:2:7'));

  assertStackFrame('Url containing parentheses',
      '    at Object.assert (http://bar:4000/bar.js?value=(a):150:3)',
      new webdriver.stacktrace.Frame('Object', 'assert', '',
          'http://bar:4000/bar.js?value=(a):150:3'));

  assertStackFrame('Frame with non-standard leading whitespace (issue 7994)',
      '  at module.exports.runCucumber (/local/dir/path)',
      new webdriver.stacktrace.Frame('module.exports', 'runCucumber', '',
          '/local/dir/path'));
}

function testParseStackFrameInOpera() {
  assertStackFrame('empty frame', '@', webdriver.stacktrace.ANONYMOUS_FRAME_);

  assertStackFrame('javascript path only',
      '@javascript:console.log(Error().stack):1',
      new webdriver.stacktrace.Frame('', '', '',
          'javascript:console.log(Error().stack):1'));

  assertStackFrame('path only',
      '@file:///foo:42',
      new webdriver.stacktrace.Frame('', '', '', 'file:///foo:42'));

  // (function go() { throw Error() })()
  // var c = go; c()
  assertStackFrame('name and empty path',
      'go([arguments not available])@',
      new webdriver.stacktrace.Frame('', 'go', '', ''));

  assertStackFrame('name and path',
      'go([arguments not available])@file:///foo:42',
      new webdriver.stacktrace.Frame('', 'go', '', 'file:///foo:42'));

  // (function() { throw Error() })()
  assertStackFrame('anonymous function',
      '<anonymous function>([arguments not available])@file:///foo:42',
      new webdriver.stacktrace.Frame('', '', '', 'file:///foo:42'));

  // var b = {foo: function() { throw Error() }}
  assertStackFrame('object literal function',
      '<anonymous function: foo>()@file:///foo:42',
      new webdriver.stacktrace.Frame('', 'foo', '', 'file:///foo:42'));

  // var c = {}; c.foo = function() { throw Error() }
  assertStackFrame('named object literal function',
      '<anonymous function: c.foo>()@file:///foo:42',
      new webdriver.stacktrace.Frame('c', 'foo', '', 'file:///foo:42'));

  assertStackFrame('prototype function',
      '<anonymous function: Foo.prototype.bar>()@',
      new webdriver.stacktrace.Frame('Foo.prototype', 'bar', '', ''));

  assertStackFrame('namespaced prototype function',
      '<anonymous function: goog.Foo.prototype.bar>()@',
      new webdriver.stacktrace.Frame(
          'goog.Foo.prototype', 'bar', '', ''));
}

function testParseClosureCanonicalStackFrame() {
  assertStackFrame('unknown frame', '> (unknown)',
      webdriver.stacktrace.ANONYMOUS_FRAME_);
  assertStackFrame('anonymous frame', '> anonymous',
      webdriver.stacktrace.ANONYMOUS_FRAME_);

  assertStackFrame('name only', '> foo',
      new webdriver.stacktrace.Frame('', 'foo', '', ''));

  assertStackFrame('name and path', '> foo at http://x:123',
      new webdriver.stacktrace.Frame('', 'foo', '', 'http://x:123'));

  assertStackFrame('anonymous function with path',
      '> anonymous at file:///x/y/z',
      new webdriver.stacktrace.Frame('', 'anonymous', '', 'file:///x/y/z'));

  assertStackFrame('anonymous function with v8 path',
      '> anonymous at /x/y/z:12:34',
      new webdriver.stacktrace.Frame('', 'anonymous', '', '/x/y/z:12:34'));

  assertStackFrame('context and name only',
      '> foo.bar', new webdriver.stacktrace.Frame('foo', 'bar', '', ''));

  assertStackFrame('name and alias',
      '> foo [as bar]', new webdriver.stacktrace.Frame('', 'foo', 'bar', ''));

  assertStackFrame('context, name, and alias',
      '> foo.bar [as baz]',
      new webdriver.stacktrace.Frame('foo', 'bar', 'baz', ''));

  assertStackFrame('path only', '> http://x:123',
      new webdriver.stacktrace.Frame('', '', '', 'http://x:123'));

  assertStackFrame('name and arguments',
      '> foo(arguments)', new webdriver.stacktrace.Frame('', 'foo', '', ''));

  assertStackFrame('full frame',
      '> foo.bar(123, "abc") [as baz] at http://x:123',
      new webdriver.stacktrace.Frame('foo', 'bar', 'baz', 'http://x:123'));

  assertStackFrame('name and url with sub-domain',
      '> foo at http://x.y.z:80/path:1:2',
      new webdriver.stacktrace.Frame('', 'foo', '',
          'http://x.y.z:80/path:1:2'));

  assertStackFrame('name and url with sub-domain',
      '> foo.bar.baz at http://x.y.z:80/path:1:2',
      new webdriver.stacktrace.Frame('foo.bar', 'baz', '',
          'http://x.y.z:80/path:1:2'));
}

// All test strings are parsed with the conventional and long
// frame algorithms.
function testParseStackFrameInFirefox() {
  var frameString = 'Error("Assertion failed")@:0';
  var frame = webdriver.stacktrace.parseStackFrame_(frameString);
  var expected = new webdriver.stacktrace.Frame('', 'Error', '', '');
  assertObjectEquals('function name + arguments', expected, frame);

  frame = webdriver.stacktrace.parseLongFirefoxFrame_(frameString);
  assertObjectEquals('function name + arguments', expected, frame);

  frameString = '()@file:///foo:42';
  frame = webdriver.stacktrace.parseStackFrame_(frameString);
  expected = new webdriver.stacktrace.Frame('', '', '', 'file:///foo:42');
  assertObjectEquals('anonymous function', expected, frame);

  frame = webdriver.stacktrace.parseLongFirefoxFrame_(frameString);
  assertObjectEquals('anonymous function', expected, frame);

  frameString = '@javascript:alert(0)';
  frame = webdriver.stacktrace.parseStackFrame_(frameString);
  expected = new webdriver.stacktrace.Frame('', '', '', 'javascript:alert(0)');
  assertObjectEquals('anonymous function', expected, frame);

  frame = webdriver.stacktrace.parseLongFirefoxFrame_(frameString);
  assertObjectEquals('anonymous function', expected, frame);
}

function testStringRepresentation() {
  var frame = new webdriver.stacktrace.Frame('window', 'foo', 'bar',
      'http://x?a=1&b=2:1');
  assertEquals('    at window.foo [as bar] (http://x?a=1&b=2:1)',
      frame.toString());

  frame = new webdriver.stacktrace.Frame('', 'Error', '', '');
  assertEquals('    at Error (<anonymous>)', frame.toString());

  assertEquals('    at <anonymous>',
      webdriver.stacktrace.ANONYMOUS_FRAME_.toString());

  frame = new webdriver.stacktrace.Frame('', '', '', 'http://x:123');
  assertEquals('    at http://x:123', frame.toString());

  frame = new webdriver.stacktrace.Frame('foo', 'bar', '', 'http://x:123');
  assertEquals('    at foo.bar (http://x:123)', frame.toString());

  frame = new webdriver.stacktrace.Frame('new ', 'Foo', '', 'http://x:123');
  assertEquals('    at new Foo (http://x:123)', frame.toString());

  frame = new webdriver.stacktrace.Frame('new foo', 'Bar', '', '');
  assertEquals('    at new foo.Bar (<anonymous>)', frame.toString());
}

// Create a stack trace string with one modest record and one long record,
// Verify that all frames are parsed. The length of the long arg is set
// to blow Firefox 3x's stack if put through a RegExp.
function testParsingLongStackTrace() {
  var longArg = goog.string.buildString(
      '(', goog.string.repeat('x', 1000000), ')');
  var stackTrace = goog.string.buildString(
      'shortFrame()@:0\n',
      'longFrame',
      longArg,
      '@http://google.com/somescript:0\n');
  var frames = webdriver.stacktrace.parse_(stackTrace);
  assertEquals('number of returned frames', 2, frames.length);
  var expected = new webdriver.stacktrace.Frame('', 'shortFrame', '', '');
  assertStackFrame('short frame', frames[0], expected);

  expected = new webdriver.stacktrace.Frame(
      '', 'longFrame', '', 'http://google.com/somescript:0');
  assertStackFrame('exception name only', frames[1], expected);
}

function testRemovesV8MessageHeaderBeforeParsingStack() {
  var stack =
        '    at Color.red (http://x:1234)\n' +
        '    at Foo.bar (http://y:5678)';

  var frames = webdriver.stacktrace.parse_(stack);
  assertEquals(2, frames.length);
  assertEquals('    at Color.red (http://x:1234)', frames[0].toString());
  assertEquals('    at Foo.bar (http://y:5678)', frames[1].toString());
}

function testCanParseClosureJsUnitExceptions() {
  stubs.set(goog.testing.stacktrace, 'get', function() {
    return '> Color.red at http://x:1234\n' +
           '> Foo.bar at http://y:5678';
  });

  var error = new goog.testing.JsUnitException('stub');
  stubs.reset();

  var frames = webdriver.stacktrace.parse_(error.stackTrace);
  assertEquals(2, frames.length);
  assertEquals('    at Color.red (http://x:1234)', frames[0].toString());
  assertEquals('    at Foo.bar (http://y:5678)', frames[1].toString());
}

function testFormattingAV8StyleError() {
  var errorStub = {
    name: 'Error',
    message: 'foo',
    toString: function() { return 'Error: foo'; },
    stack:
        'Error: foo\n' +
        '    at Color.red (http://x:1234)\n' +
        '    at Foo.bar (http://y:5678)'
  };

  var ret = webdriver.stacktrace.format(errorStub);
  assertEquals(errorStub, ret);
  assertEquals([
    'Error: foo',
    '    at Color.red (http://x:1234)',
    '    at Foo.bar (http://y:5678)'
  ].join('\n'), ret.stack);
}

function testFormattingAFirefoxStyleError() {
  var errorStub = {
    name: 'Error',
    message: 'boom',
    toString: function() { return 'Error: boom'; },
    stack:
        'foo@file:///foo/foo.js:1\n' +
        '@file:///bar/bar.js:1'
  };

  var ret = webdriver.stacktrace.format(errorStub);
  assertEquals(errorStub, ret);
  assertEquals([
    'Error: boom',
    '    at foo (file:///foo/foo.js:1)',
    '    at file:///bar/bar.js:1'
  ].join('\n'), ret.stack);
}

function testInsertsAnAnonymousFrameWhenUnableToParse() {
  var errorStub = {
    name: 'Error',
    message: 'boom',
    toString: function() { return 'Error: boom'; },
    stack:
        'foo@file:///foo/foo.js:1\n' +
        'this is unparsable garbage\n' +
        '@file:///bar/bar.js:1'
  };

  var ret = webdriver.stacktrace.format(errorStub);
  assertEquals(errorStub, ret);
  assertEquals([
    'Error: boom',
    '    at foo (file:///foo/foo.js:1)',
    '    at <anonymous>',
    '    at file:///bar/bar.js:1'
  ].join('\n'), ret.stack);
}

function testFormattingBotErrors() {
  var error = new bot.Error(bot.ErrorCode.NO_SUCH_ELEMENT, 'boom');
  var expectedStack = [
    'NoSuchElementError: boom',
    '    at Color.red (http://x:1234)',
    '    at Foo.bar (http://y:5678)'
  ].join('\n');
  error.stack = expectedStack;

  var ret = webdriver.stacktrace.format(error);
  assertEquals(ret, error);
  assertEquals(expectedStack, error.stack);
}

function testFormatsUsingNameAndMessageIfAvailable() {
  var ret = webdriver.stacktrace.format({
    name: 'TypeError',
    message: 'boom'
  });
  assertEquals('TypeError: boom\n', ret.stack);

  ret = webdriver.stacktrace.format({
    message: 'boom'
  });
  assertEquals('boom\n', ret.stack);

  ret = webdriver.stacktrace.format({
    toString: function() {
      return 'Hello world'
    }
  });
  assertEquals('Hello world\n', ret.stack);

  ret = webdriver.stacktrace.format({
    name: 'TypeError',
    message: 'boom',
    toString: function() {
      return 'Should not use this'
    }
  });
  assertEquals('TypeError: boom\n', ret.stack);
}

function testDoesNotFormatErrorIfOriginalStacktraceIsInAnUnexpectedFormat() {
  var error = Error('testing');
  var stack = error.stack = [
    'Error: testing',
    '..> at Color.red (http://x:1234)',
    '..> at Foo.bar (http://y:5678)'
  ].join('\n');

  var ret = webdriver.stacktrace.format(error);
  assertEquals(ret, error);
  assertEquals(stack, error.stack);
}

function testParseStackFrameInIE10() {
  assertStackFrame('name and path',
      '   at foo (http://bar:4000/bar.js:150:3)',
      new webdriver.stacktrace.Frame('', 'foo', '',
          'http://bar:4000/bar.js:150:3'));

  assertStackFrame('Anonymous function',
      '   at Anonymous function (http://bar:4000/bar.js:150:3)',
      new webdriver.stacktrace.Frame('', 'Anonymous function', '',
          'http://bar:4000/bar.js:150:3'));

  assertStackFrame('Global code',
      '   at Global code (http://bar:4000/bar.js:150:3)',
      new webdriver.stacktrace.Frame('', 'Global code', '',
          'http://bar:4000/bar.js:150:3'));

  assertStackFrame('eval code',
      '   at foo (eval code:150:3)',
      new webdriver.stacktrace.Frame('', 'foo', '', 'eval code:150:3'));

  assertStackFrame('nested eval',
      '   at eval code (eval code:150:3)',
      new webdriver.stacktrace.Frame('', 'eval code', '', 'eval code:150:3'));

  assertStackFrame('Url containing parentheses',
      '   at foo (http://bar:4000/bar.js?value=(a):150:3)',
      new webdriver.stacktrace.Frame('', 'foo', '',
          'http://bar:4000/bar.js?value=(a):150:3'));

  assertStackFrame('Url ending with parentheses',
      '   at foo (http://bar:4000/bar.js?a=))',
      new webdriver.stacktrace.Frame('', 'foo', '',
          'http://bar:4000/bar.js?a=)'));
}
