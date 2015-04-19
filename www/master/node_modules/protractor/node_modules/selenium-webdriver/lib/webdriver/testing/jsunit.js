// Copyright 2011 Software Freedom Conservancy. All Rights Reserved.
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

/**
 * @fileoverview File to include for turning any HTML file page into a WebDriver
 * JSUnit test suite by configuring an onload listener to the body that will
 * instantiate and start the test runner.
 */

goog.provide('webdriver.testing.jsunit');
goog.provide('webdriver.testing.jsunit.TestRunner');

goog.require('goog.testing.TestRunner');
goog.require('webdriver.testing.Client');
goog.require('webdriver.testing.TestCase');



/**
 * Constructs a test runner.
 * @param {!webdriver.testing.Client} client .
 * @constructor
 * @extends {goog.testing.TestRunner}
 */
webdriver.testing.jsunit.TestRunner = function(client) {
  goog.base(this);

  /** @private {!webdriver.testing.Client} */
  this.client_ = client;
};
goog.inherits(webdriver.testing.jsunit.TestRunner, goog.testing.TestRunner);


/**
 * Element created in the document to add test results to.
 * @private {Element}
 */
webdriver.testing.jsunit.TestRunner.prototype.logEl_ = null;


/**
 * DOM element used to stored screenshots. Screenshots are stored in the DOM to
 * avoid exhausting JS stack-space.
 * @private {Element}
 */
webdriver.testing.jsunit.TestRunner.prototype.screenshotCacheEl_ = null;


/** @override */
webdriver.testing.jsunit.TestRunner.prototype.initialize = function(testCase) {
  goog.base(this, 'initialize', testCase);
  this.screenshotCacheEl_ = document.createElement('div');
  document.body.appendChild(this.screenshotCacheEl_);
  this.screenshotCacheEl_.style.display = 'none';
};


/** @override */
webdriver.testing.jsunit.TestRunner.prototype.execute = function() {
  if (!this.testCase) {
    throw Error('The test runner must be initialized with a test case before ' +
                'execute can be called.');
  }
  this.screenshotCacheEl_.innerHTML = '';
  this.client_.sendInitEvent();
  this.testCase.setCompletedCallback(goog.bind(this.onComplete_, this));
  this.testCase.runTests();
};


/**
 * Writes a nicely formatted log out to the document. Overrides
 * {@link goog.testing.TestRunner#writeLog} to handle writing screenshots to the
 *     log.
 * @param {string} log The string to write.
 * @override
 */
webdriver.testing.jsunit.TestRunner.prototype.writeLog = function(log) {
  var lines = log.split('\n');
  for (var i = 0; i < lines.length; i++) {
    var line = lines[i];
    var color;
    var isFailOrError = /FAILED/.test(line) || /ERROR/.test(line);
    var isScreenshot = / \[SCREENSHOT\] /.test(line);
    if (/PASSED/.test(line)) {
      color = 'darkgreen';
    } else if (isFailOrError) {
      color = 'darkred';
    } else if (isScreenshot) {
      color = 'darkblue';
    } else {
      color = '#333';
    }

    var div = document.createElement('div');
    if (line.substr(0, 2) == '> ') {
      // The stack trace may contain links so it has to be interpreted as HTML.
      div.innerHTML = line;
    } else {
      div.appendChild(document.createTextNode(line));
    }

    if (isFailOrError) {
      var testNameMatch = /(\S+) (\[[^\]]*] )?: (FAILED|ERROR)/.exec(line);
      if (testNameMatch) {
        // Build a URL to run the test individually.  If this test was already
        // part of another subset test, we need to overwrite the old runTests
        // query parameter.  We also need to do this without bringing in any
        // extra dependencies, otherwise we could mask missing dependency bugs.
        var newSearch = 'runTests=' + testNameMatch[1];
        var search = window.location.search;
        if (search) {
          var oldTests = /runTests=([^&]*)/.exec(search);
          if (oldTests) {
            newSearch = search.substr(0, oldTests.index) +
                        newSearch +
                        search.substr(oldTests.index + oldTests[0].length);
          } else {
            newSearch = search + '&' + newSearch;
          }
        } else {
          newSearch = '?' + newSearch;
        }
        var href = window.location.href;
        var hash = window.location.hash;
        if (hash && hash.charAt(0) != '#') {
          hash = '#' + hash;
        }
        href = href.split('#')[0].split('?')[0] + newSearch + hash;

        // Add the link.
        var a = document.createElement('A');
        a.innerHTML = '(run individually)';
        a.style.fontSize = '0.8em';
        a.href = href;
        div.appendChild(document.createTextNode(' '));
        div.appendChild(a);
      }
    }

    if (isScreenshot && this.screenshotCacheEl_.childNodes.length) {
      var nextScreenshot = this.screenshotCacheEl_.childNodes[0];
      this.screenshotCacheEl_.removeChild(nextScreenshot);

      a = document.createElement('A');
      a.style.fontSize = '0.8em';
      a.href = 'javascript:void(0);';
      a.onclick = goog.partial(toggleVisibility, a, nextScreenshot);
      toggleVisibility(a, nextScreenshot);
      div.appendChild(document.createTextNode(' '));
      div.appendChild(a);
    }

    div.style.color = color;
    div.style.font = 'normal 100% monospace';

    try {
      div.style.whiteSpace = 'pre-wrap';
    } catch (e) {
      // NOTE(user): IE raises an exception when assigning to pre-wrap.
      // Thankfully, it doesn't collapse whitespace when using monospace fonts,
      // so it will display correctly if we ignore the exception.
    }

    if (i < 2) {
      div.style.fontWeight = 'bold';
    }
    this.logEl_.appendChild(div);

    if (nextScreenshot) {
      a = document.createElement('A');
      // Accessing the |src| property in IE sometimes results in an
      // "Invalid pointer" error, which indicates it has been garbage
      // collected. This does not occur when using getAttribute.
      a.href = nextScreenshot.getAttribute('src');
      a.target = '_blank';
      a.appendChild(nextScreenshot);
      this.logEl_.appendChild(a);
      nextScreenshot = null;
    }
  }

  function toggleVisibility(link, img) {
    if (img.style.display === 'none') {
      img.style.display = '';
      link.innerHTML = '(hide screenshot)';
    } else {
      img.style.display = 'none';
      link.innerHTML = '(view screenshot)';
    }
  }
};


/**
 * Copied from goog.testing.TestRunner.prototype.onComplete_, which has private
 * visibility.
 * @private
 */
webdriver.testing.jsunit.TestRunner.prototype.onComplete_ = function() {
  var log = this.testCase.getReport(true);
  if (this.errors.length > 0) {
    log += '\n' + this.errors.join('\n');
  }

  if (!this.logEl_) {
    this.logEl_ = document.createElement('div');
    document.body.appendChild(this.logEl_);
  }

  // Remove all children from the log element.
  var logEl = this.logEl_;
  while (logEl.firstChild) {
    logEl.removeChild(logEl.firstChild);
  }

  this.writeLog(log);
  this.client_.sendResultsEvent(this.isSuccess(), this.getReport(true));
};


/**
 * Takes a screenshot. In addition to saving the screenshot for viewing in the
 * HTML logs, the screenshot will also be saved using
 * @param {!webdriver.WebDriver} driver The driver to take the screenshot with.
 * @param {string=} opt_label An optional debug label to identify the screenshot
 *     with.
 * @return {!webdriver.promise.Promise} A promise that will be resolved to the
 *     screenshot as a base-64 encoded PNG.
 */
webdriver.testing.jsunit.TestRunner.prototype.takeScreenshot = function(
    driver, opt_label) {
  if (!this.isInitialized()) {
    throw Error(
        'The test runner must be initialized before it may be used to' +
        ' take screenshots');
  }

  var client = this.client_;
  var testCase = this.testCase;
  var screenshotCache = this.screenshotCacheEl_;
  return driver.takeScreenshot().then(function(png) {
    client.sendScreenshotEvent(png, opt_label);

    var img = document.createElement('img');
    img.src = 'data:image/png;base64,' + png;
    img.style.border = '1px solid black';
    img.style.maxWidth = '500px';
    screenshotCache.appendChild(img);

    if (testCase) {
      testCase.saveMessage('[SCREENSHOT] ' + (opt_label || '<Not Labeled>'));
    }
    return png;
  });
};


/**
 * Sends a base64 encoded PNG image to the server to be saved in the test
 * outputs.
 * @param {string} data The base64 encoded PNG image to be sent to the server.
 * @param {string=} opt_label An optional debug label to identify the
 *     screenshot with.
 */
webdriver.testing.jsunit.TestRunner.prototype.saveImage = function(
    data, opt_label) {
  if (!this.isInitialized()) {
    throw Error(
        'The test runner must be initialized before it may be used to' +
        ' save images');
  }

  this.client_.sendScreenshotEvent(data, opt_label);

  var img = document.createElement('img');
  img.src = 'data:image/png;base64,' + data;
  img.style.border = '1px solid black';
  img.style.maxWidth = '500px';
  this.screenshotCacheEl_.appendChild(img);

  if (this.testCase) {
    this.testCase.saveMessage('[SCREENSHOT] ' + (opt_label || '<Not Labeled>'));
  }
};


(function() {
  var client = new webdriver.testing.Client();
  var tr = new webdriver.testing.jsunit.TestRunner(client);

  // Export our test runner so it can be accessed by Selenium/WebDriver. This
  // will only work if webdriver.WebDriver is using a pure-JavaScript
  // webdriver.CommandExecutor. Otherwise, the JS-client could change the
  // driver's focus to another window or frame and the Java/Python-client
  // wouldn't be able to access this object.
  goog.exportSymbol('G_testRunner', tr);
  goog.exportSymbol('G_testRunner.initialize', tr.initialize);
  goog.exportSymbol('G_testRunner.isInitialized', tr.isInitialized);
  goog.exportSymbol('G_testRunner.isFinished', tr.isFinished);
  goog.exportSymbol('G_testRunner.isSuccess', tr.isSuccess);
  goog.exportSymbol('G_testRunner.getReport', tr.getReport);
  goog.exportSymbol('G_testRunner.getRunTime', tr.getRunTime);
  goog.exportSymbol('G_testRunner.getNumFilesLoaded', tr.getNumFilesLoaded);
  goog.exportSymbol('G_testRunner.setStrict', tr.setStrict);
  goog.exportSymbol('G_testRunner.logTestFailure', tr.logTestFailure);

  // Export debug as a global function for JSUnit compatibility.  This just
  // calls log on the current test case.
  if (!goog.global['debug']) {
    goog.exportSymbol('debug', goog.bind(tr.log, tr));
  }

  // Add an error handler to report errors that may occur during
  // initialization of the page.
  var onerror = window.onerror;
  window.onerror = function(error, url, line) {
    // Call any existing onerror handlers.
    if (onerror) {
      onerror(error, url, line);
    }
    if (typeof error == 'object') {
      // Webkit started passing an event object as the only argument to
      // window.onerror.  It doesn't contain an error message, url or line
      // number.  We therefore log as much info as we can.
      if (error.target && error.target.tagName == 'SCRIPT') {
        tr.logError('UNKNOWN ERROR: Script ' + error.target.src);
      } else {
        tr.logError('UNKNOWN ERROR: No error information available.');
      }
    } else {
      tr.logError('JS ERROR: ' + error + '\nURL: ' + url + '\nLine: ' + line);
    }
  };

  var onload = window.onload;
  window.onload = function() {
    // Call any existing onload handlers.
    if (onload) {
      onload();
    }

    var testCase = new webdriver.testing.TestCase(client, document.title);
    testCase.autoDiscoverTests();

    tr.initialize(testCase);
    tr.execute();
  };
})();
