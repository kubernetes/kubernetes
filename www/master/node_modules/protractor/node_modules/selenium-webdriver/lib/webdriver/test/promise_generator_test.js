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

goog.provide('webdriver.test.promise.generator.test');
goog.setTestOnly('webdriver.test.promise.generator.test');

goog.require('goog.testing.AsyncTestCase');
goog.require('goog.testing.jsunit');
goog.require('webdriver.promise');


var test = goog.testing.AsyncTestCase.createAndInstall(
    'promise_generator_test');


function testRequiresInputsToBeGeneratorFunctions() {
  var thrown = assertThrows(function() {
    webdriver.promise.consume(function() {});
  });
  assertTrue(thrown instanceof TypeError);
}


function testBasicGenerator() {
  var values = [];
  test.waitForAsync();
  webdriver.promise.consume(function* () {
    var i = 0;
    while (i < 4) {
      i = yield i + 1;
      values.push(i);
    }
  }).then(function() {
    test.continueTesting();
    assertArrayEquals([1, 2, 3, 4], values);
  });
}


function testPromiseYieldingGenerator() {
  var values = [];
  webdriver.promise.consume(function* () {
    var i = 0;
    while (i < 4) {
      // Test that things are actually async here.
      setTimeout(function() {
        values.push(i * 2);
      }, 10);

      yield webdriver.promise.delayed(10).then(function() {
        values.push(i++);
      });
    }
  }).then(function() {
    test.continueTesting();
    assertArrayEquals([0, 0, 2, 1, 4, 2, 6, 3], values);
  });
  test.waitForAsync();
}


function testAssignmentsToYieldedPromisesGetFulfilledValue() {
  test.waitForAsync();
  webdriver.promise.consume(function* () {
    var p = webdriver.promise.fulfilled(2);
    var x = yield p;
    assertEquals(2, x);
  }).then(function() {
    test.continueTesting();
  });
}


function testCanCancelPromiseGenerator() {
  var values = [];
  var p = webdriver.promise.consume(function* () {
    var i = 0;
    while (i < 3) {
      yield webdriver.promise.delayed(100).then(function() {
        values.push(i++);
      });
    }
  });
  setTimeout(function() {
    test.waitForAsync('cancelled; verifying it took');
    p.cancel();
    p.thenCatch(function() {
      setTimeout(function() {
        assertArrayEquals([0], values);
        test.continueTesting();
      }, 300);
    });
  }, 75);
  test.waitForAsync();
}


function testFinalReturnValueIsUsedAsFulfillmentValue() {
  test.waitForAsync();
  webdriver.promise.consume(function* () {
    yield 1;
    yield 2;
    return 3;
  }).then(function(value) {
    assertEquals(3, value);
    test.continueTesting();
  });
}


function testRejectionsAreThrownWithinGenerator() {
  test.waitForAsync();
  var values = [];
  webdriver.promise.consume(function* () {
    values.push('a');
    var e = Error('stub error');
    try {
      yield webdriver.promise.rejected(e);
      values.push('b');
    } catch (ex) {
      assertEquals(e, ex);
      values.push('c');
    }
    values.push('d');
  }).then(function() {
    assertArrayEquals(['a', 'c', 'd'], values);
    test.continueTesting();
  });
}


function testUnhandledRejectionsAbortGenerator() {
  test.waitForAsync();

  var values = [];
  var e = Error('stub error');
  webdriver.promise.consume(function* () {
    values.push(1);
    yield webdriver.promise.rejected(e);
    values.push(2);
  }).thenCatch(function() {
    assertArrayEquals([1], values);
    test.continueTesting();
  });
}


function testYieldsWaitForPromises() {
  test.waitForAsync();

  var values = [];
  var d = webdriver.promise.defer();
  webdriver.promise.consume(function* () {
    values.push(1);
    values.push((yield d.promise), 3);
  }).then(function() {
    assertArrayEquals([1, 2, 3], values);
    test.continueTesting();
  });

  setTimeout(function() {
    assertArrayEquals([1], values);
    d.fulfill(2);
  }, 100);
}


function testCanSpecifyGeneratorScope() {
  test.waitForAsync();
  webdriver.promise.consume(function* () {
    return this.name;
  }, {name: 'Bob'}).then(function(value) {
    assertEquals('Bob', value);
    test.continueTesting();
  });
}


function testCanSpecifyGeneratorArgs() {
  test.waitForAsync();
  webdriver.promise.consume(function* (a, b) {
    assertEquals('red', a);
    assertEquals('apples', b);
  }, null, 'red', 'apples').then(function() {
    test.continueTesting();
  });
}


function testExecuteGeneratorInAFlow() {
  var promises = [
      webdriver.promise.defer(),
      webdriver.promise.defer()
  ];
  var values = [];
  webdriver.promise.controlFlow().execute(function* () {
    values.push(yield promises[0].promise);
    values.push(yield promises[1].promise);
    values.push('fin');
  }).then(function() {
    assertArrayEquals([1, 2, 'fin'], values);
    test.continueTesting();
  });

  test.waitForAsync();
  setTimeout(function() {
    assertArrayEquals([], values);
    promises[0].fulfill(1);
  }, 100);
  setTimeout(function() {
    assertArrayEquals([1], values);
    promises[1].fulfill(2);
  }, 200);
}


function testNestedGeneratorsInAFlow() {
  var flow = webdriver.promise.controlFlow();
  flow.execute(function* () {
    var x = yield flow.execute(function() {
      return webdriver.promise.delayed(10).then(function() {
        return 1;
      });
    });

    var y = yield flow.execute(function() {
      return 2;
    });

    return x + y;
  }).then(function(value) {
    assertEquals(3, value);
    test.continueTesting();
  });
  test.waitForAsync();
}


function testFlowWaitOnGenerator() {
  var values = [];
  webdriver.promise.controlFlow().wait(function* () {
    yield values.push(1);
    values.push(yield webdriver.promise.delayed(10).then(function() {
      return 2;
    }));
    yield values.push(3);
    return values.length === 6;
  }, 250).then(function() {
    assertArrayEquals([1, 2, 3, 1, 2, 3], values);
    test.continueTesting();
  });
  test.waitForAsync();
}


function testFlowWaitingOnGeneratorTimesOut() {
  var values = [];
  webdriver.promise.controlFlow().wait(function* () {
    var i = 0;
    while (i < 3) {
      yield webdriver.promise.delayed(100).then(function() {
        values.push(i++);
      });
    }
  }, 75).thenCatch(function() {
    assertArrayEquals('Should complete one loop of wait condition',
        [0, 1, 2], values);
    test.continueTesting();
  });
  test.waitForAsync();
}

