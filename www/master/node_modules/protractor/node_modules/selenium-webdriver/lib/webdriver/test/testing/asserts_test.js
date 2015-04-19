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

goog.require('goog.testing.jsunit');
goog.require('webdriver.test.testutil');
goog.require('webdriver.testing.assert');
goog.require('webdriver.testing.asserts');

var assert = webdriver.testing.assert;
var result;

function setUp() {
  result = webdriver.test.testutil.callbackPair();
}


function testAssertion_nonPromiseValue_valueMatches() {
  assert('foo').equalTo('foo');
  // OK if it does not throw.
}


function testAssertion_nonPromiseValue_notValueMatches() {
  var a = assert('foo');
  assertThrows(goog.bind(a.equalTo, a, 'bar'));
}


function testAssertion_promiseValue_valueMatches() {
  var d = new webdriver.promise.Deferred();
  assert(d).equalTo('foo').then(result.callback, result.errback);
  result.assertNeither();
  d.fulfill('foo');
  result.assertCallback();
}


function testAssertion_promiseValue_notValueMatches() {
  var d = new webdriver.promise.Deferred();
  assert(d).equalTo('foo').then(result.callback, result.errback);
  result.assertNeither();
  d.fulfill('bar');
  result.assertErrback();
}


function testAssertion_promiseValue_promiseRejected() {
  var d = new webdriver.promise.Deferred();
  assert(d).equalTo('foo').then(result.callback, result.errback);
  result.assertNeither();
  d.reject();
  result.assertErrback();
}


function testAssertion_decoration() {
  assert('foo').is.equalTo('foo');
  // Ok if no throws.
}


function testAssertion_negation() {
  var a = assert('foo');

  a.not.equalTo('bar');  // OK if this does not throw.
  a.is.not.equalTo('bar');  // OK if this does not throw.

  var notA = a.not;
  assertThrows(goog.bind(notA.equalTo, notA, 123));

  notA = a.is.not;
  assertThrows(goog.bind(notA.equalTo, notA, 123));
}


function testApplyMatcher_nonPromiseValue_valueMatches() {
  assertThat('foo', equals('foo')).
      then(result.callback, result.errback);
  result.assertCallback();
}


function testApplyMatcher_nonPromiseValue_notValueMatches() {
  assertThat('foo', equals('bar')).
      then(result.callback, result.errback);
  result.assertErrback();
}


function testApplyMatcher_promiseValue_valueMatches() {
  var d = new webdriver.promise.Deferred();
  assertThat(d, equals('foo')).
      then(result.callback, result.errback);
  result.assertNeither();
  d.fulfill('foo');
  result.assertCallback();
}


function testApplyMatcher_promiseValue_notValueMatches() {
  var d = new webdriver.promise.Deferred();
  assertThat(d, equals('foo')).
      then(result.callback, result.errback);
  result.assertNeither();
  d.fulfill('bar');
  result.assertErrback();
}


function testApplyMatcher_promiseValue_promiseRejected() {
  var d = new webdriver.promise.Deferred();
  assertThat(d, equals('foo')).
      then(result.callback, result.errback);
  result.assertNeither();
  d.reject();
  result.assertErrback();
}
