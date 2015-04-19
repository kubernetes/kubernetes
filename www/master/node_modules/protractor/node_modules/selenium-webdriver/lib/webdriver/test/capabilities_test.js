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
goog.require('webdriver.Capabilities');

function testSettingAndUnsettingACapability() {
  var caps = new webdriver.Capabilities();
  assertNull(caps.get('foo'));

  caps.set('foo', 'bar');
  assertEquals('bar', caps.get('foo'));

  caps.set('foo', null);
  assertNull(caps.get('foo'));
}


function testCheckingIfACapabilityIsSet() {
  var caps = new webdriver.Capabilities();
  assertFalse(caps.has('foo'));
  assertNull(caps.get('foo'));

  caps.set('foo', 'bar');
  assertTrue(caps.has('foo'));

  caps.set('foo', true);
  assertTrue(caps.has('foo'));

  caps.set('foo', false);
  assertFalse(caps.has('foo'));
  assertFalse(caps.get('foo'));

  caps.set('foo', null);
  assertFalse(caps.has('foo'));
  assertNull(caps.get('foo'));
}


function testMergingCapabilities() {
  var caps1 = new webdriver.Capabilities().
      set('foo', 'bar').
      set('color', 'red');

  var caps2 = new webdriver.Capabilities().
      set('color', 'green');

  assertEquals('bar', caps1.get('foo'));
  assertEquals('red', caps1.get('color'));
  assertEquals('green', caps2.get('color'));
  assertNull(caps2.get('foo'));

  caps2.merge(caps1);
  assertEquals('bar', caps1.get('foo'));
  assertEquals('red', caps1.get('color'));
  assertEquals('bar', caps2.get('foo'));
  assertEquals('red', caps2.get('color'));
}
