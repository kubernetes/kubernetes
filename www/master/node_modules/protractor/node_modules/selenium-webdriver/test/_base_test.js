// Copyright 2014 Software Freedom Conservancy. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

var assert = require('assert'),
    fs = require('fs'),
    path = require('path');

var base = require('../_base');

describe('Context', function() {
  it('does not pollute the global scope', function() {
    assert.equal('undefined', typeof goog);

    var context = new base.Context();
    assert.equal('undefined', typeof goog);
    assert.equal('object', typeof context.closure.goog);

    context.closure.goog.require('goog.array');
    assert.equal('undefined', typeof goog);
    assert.equal('object', typeof context.closure.goog.array);
  });
});


function haveGenerators() {
  try {
    // Test for generator support.
    new Function('function* x() {}');
    return true;
  } catch (ex) {
    return false;
  }
}


function runClosureTest(file) {
  var name = path.basename(file);
  name = name.substring(0, name.length - '.js'.length);

  // Generator tests will fail to parse in ES5, so mark those tests as
  // pending under ES5.
  if (name.indexOf('_generator_') != -1 && !haveGenerators()) {
    it(name);
    return;
  }

  describe(name, function() {
    var context = new base.Context(true);
    context.closure.document.title = name;
    if (process.env.VERBOSE != '1') {
      // Null out console so everything loads silently.
      context.closure.console = null;
    }
    context.closure.CLOSURE_IMPORT_SCRIPT(file);

    var tc = context.closure.G_testRunner.testCase;
    if (!tc) {
      tc = new context.closure.goog.testing.TestCase(name);
      tc.autoDiscoverTests();
    }

    var shouldRunTests = tc.shouldRunTests();
    var allTests = tc.getTests();
    allTests.forEach(function(test) {
      if (!shouldRunTests) {
        it(test.name);
        return;
      }

      it(test.name, function(done) {
        tc.setTests([test]);
        tc.setCompletedCallback(function() {
          if (tc.isSuccess()) {
            return done();
          }
          var results = tc.getTestResults();
          done(Error('\n' + Object.keys(results).map(function(name) {
            var msg = [name + ': ' + (results[name].length ? 'FAILED' : 'PASSED')];
            if (results[name].length) {
              msg = msg.concat(results[name]);
            }
            return msg.join('\n');
          }).join('\n')));
        });
        tc.runTests();
      });
    });
  });
}


function findTests(dir) {
  fs.readdirSync(dir).forEach(function(name) {
    var file = path.join(dir, name);

    var stat = fs.statSync(file);
    if (stat.isDirectory() && name !== 'atoms' && name !== 'e2e') {
      findTests(file);
      return;
    }

    var l = file.length - '_test.js'.length;
    if (l >= 0 && file.indexOf('_test.js', l) == l) {
      runClosureTest(file);
    }
  });
}

findTests(path.join(
    __dirname, base.isDevMode() ? '../../..' : '../lib', 'webdriver/test'));
