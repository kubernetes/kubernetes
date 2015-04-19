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
goog.require('webdriver.EventEmitter');

function testEmittingWhenNothingIsRegistered() {
  var emitter = new webdriver.EventEmitter();
  emitter.emit('foo');
  // Ok if no errors are thrown.
}

function testPassesArgsToListenersOnEmit() {
  var emitter = new webdriver.EventEmitter();
  var now = goog.now();

  var messages = [];
  emitter.addListener('foo', function(arg) { messages.push(arg); });

  emitter.emit('foo', now);
  assertEquals(1, messages.length);
  assertEquals(now, messages[0]);

  emitter.emit('foo', now + 15);
  assertEquals(2, messages.length);
  assertEquals(now, messages[0]);
  assertEquals(now + 15, messages[1]);
}

function testAddingListeners() {
  var emitter = new webdriver.EventEmitter();
  var count = 0;
  emitter.addListener('foo', function() { count++; });
  emitter.addListener('foo', function() { count++; });
  emitter.addListener('foo', function() { count++; });
  emitter.emit('foo');
  assertEquals(3, count);
}

function testAddingOneShotListeners() {
  var emitter = new webdriver.EventEmitter();
  var count = 0;
  emitter.once('foo', function() { count++; });
  emitter.once('foo', function() { count++; });
  emitter.once('foo', function() { count++; });

  emitter.emit('foo');
  assertEquals(3, count);

  emitter.emit('foo');
  assertEquals(3, count);
}

function testCanOnlyAddAListenerConfigOnce() {
  var emitter = new webdriver.EventEmitter();
  var count = 0;
  function onFoo() { count++; }

  emitter.on('foo', onFoo);
  emitter.on('foo', onFoo);

  emitter.emit('foo');
  assertEquals(1, count);

  emitter.emit('foo');
  assertEquals(2, count);
}

function testAddingListenersWithCustomScope() {
  var obj = {
    count: 0,
    inc: function() {
      this.count++;
    }
  };
  var emitter = new webdriver.EventEmitter();
  emitter.addListener('foo', obj.inc, obj);

  emitter.emit('foo');
  assertEquals(1, obj.count);

  emitter.emit('foo');
  assertEquals(2, obj.count);

  emitter.once('bar', obj.inc, obj);
  emitter.emit('bar');
  assertEquals(3, obj.count);

  emitter.emit('bar');
  assertEquals(3, obj.count);
}

function testRemovingListeners() {
  var emitter = new webdriver.EventEmitter();
  var count = 0;
  emitter.addListener('foo', function() { count++; });
  emitter.addListener('foo', function() { count++; });

  var toRemove = function() { count++; };
  emitter.addListener('foo', toRemove);

  emitter.emit('foo');
  assertEquals(3, count);

  emitter.removeListener('foo', toRemove);
  emitter.emit('foo');
  assertEquals(5, count);
}

function testRemovingAllListeners() {
  var emitter = new webdriver.EventEmitter();
  var count = 0;
  emitter.addListener('foo', function() { count++; });
  emitter.addListener('foo', function() { count++; });
  emitter.addListener('foo', function() { count++; });

  emitter.emit('foo');
  assertEquals(3, count);

  emitter.removeAllListeners('foo');

  emitter.emit('foo');
  assertEquals(3, count);
}

function testRemovingAbsolutelyAllListeners() {
  var emitter = new webdriver.EventEmitter();
  var messages = [];
  emitter.addListener('foo', function() { messages.push('foo'); });
  emitter.addListener('bar', function() { messages.push('bar'); });

  emitter.emit('foo');
  assertArrayEquals(['foo'], messages);

  emitter.emit('bar');
  assertArrayEquals(['foo', 'bar'], messages);

  emitter.emit('bar');
  emitter.emit('foo');
  assertArrayEquals(['foo', 'bar', 'bar', 'foo'], messages);

  emitter.removeAllListeners();
  assertArrayEquals(['foo', 'bar', 'bar', 'foo'], messages);
}

function testListenerThatRemovesItself() {
  var emitter = new webdriver.EventEmitter();
  emitter.on('foo', listener);
  emitter.emit('foo');

  function listener() {
    emitter.removeListener('foo', listener);
  }
}

function testListenerThatRemovesItself_whenThereAreMultipleListeners() {
  var emitter = new webdriver.EventEmitter();
  var messages = [];
  emitter.on('foo', one);
  emitter.on('foo', two);
  emitter.emit('foo');
  assertArrayEquals(['one', 'two'], messages);

  function one() {
    messages.push('one');
    emitter.removeListener('foo', one);
  }

  function two() {
    messages.push('two');
  }
}

function testListenerAddsAnotherForTheSameEventType() {
  var emitter = new webdriver.EventEmitter();
  var messages = [];
  emitter.on('foo', one);
  emitter.emit('foo');
  assertArrayEquals(['one', 'two'], messages);

  function one() {
    messages.push('one');
    emitter.on('foo', two);
  }

  function two() {
    messages.push('two');
  }
}
