var q = require('q'),
    fs = require('fs'),
    _ = require('lodash'),
    ngHintNames = {
      ngHintControllers: 'Controllers',
      ngHintDirectives: 'Directives',
      ngHintDom: 'DOM',
      ngHintEvents: 'Events',
      ngHintInterpolation: 'Interpolation',
      ngHintModules: 'Modules',
      ngHintScopes: 'Scopes',
      ngHintGeneral: 'General'
    };

/**
 * You enable this plugin in your config file:
 *
 *    exports.config = {
 *      plugins: [{
 *        path: 'node_modules/protractor/plugins/ngHint',
 *
 *        asTests: {Boolean},
 *        excludeURLs: {(String|RegExp)[]}
 *      }]
 *    };
 *
 * asTests specifies if the plugin should generate passed/failed test results
 * based on the ngHint output or instead log the results to the console.
 * Defaults to true.
 *
 * excludeURLs specifies a list of URLs for which ngHint results should be
 * ignored.  Defaults to []
 */

/*
 * The strategy for this plugin is as follows:
 *
 * During setup, install the ngHint code and listeners to capture its output.
 * Store the output in the following format:
 *    {page URL} -> {module} -> {message} -> {message type}
 *
 * So, for instance, you might have:
 *    {
 *      'google.com': {
 *        'Controllers': {
 *          {'It is against Angular best practices to...': warning}
 *        },
 *        'Modules: {
 *          {'Module "Search" was loaded but does not exist': error}
 *        }
 *      }
 *    }
 *
 * We store the messages as keys in objects in order to avoid duplicates.
 */

/**
 * Configures the plugin to grab the output of ngHint
 *
 * @param {Object} config The configuration file for the ngHint plugin
 * @public
 */
function setup(config) {
  // Intercept ngHint output
  browser.addMockModule('protractorNgHintCaptureModule_', function() {
    angular.module('protractorNgHintCaptureModule_', []);
    var hintInstalled = true;
    if (!angular.hint) {
      hintInstalled = false;
      angular.hint = {};
    }
    /** override */
    angular.hint.onMessage = function(module, message, type) {
      var ngHintLog = JSON.parse(localStorage.getItem(
                                  'ngHintLog_protractor') || '{}');
      var pageLog = ngHintLog[location] || {};
      var moduleLog = pageLog[module] || {};
      moduleLog[message] = type;
      pageLog[module] = moduleLog;
      ngHintLog[location] = pageLog;
      localStorage.setItem('ngHintLog_protractor',
                            JSON.stringify(ngHintLog));
    };
    if (!hintInstalled) {
      angular.hint.onMessage('General', 'ngHint plugin cannot be run as ' +
          'ngHint code was never included into the page', 'warning');
    }
  });
}

/**
 * Checks if a URL should not be examined by the ngHint plugin
 *
 * @param {String} url The URL to check for exclusion
 * @param {Object} config The configuration file for the ngHint plugin
 * @return {Boolean} true if the URL should not be examined by the ngHint plugin
 * @private
 */
function isExcluded(url, config) {
  var excludeURLs = config.excludeURLs || [];
  for (var i = 0; i < excludeURLs.length; i++) {
    if (typeof excludeURLs[i] == typeof '') {
      if (url == excludeURLs[i]) {
        return true;
      }
    } else {
      if (excludeURLs[i].test(url)) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Checks if a warning message should be ignored by the ngHint plugin
 *
 * @param {String} message The message to check
 * @return {Boolean} true if the message should be ignored
 * @private
 */
function isMessageToIgnore(message) {
  if (message == 'It is against Angular best practices to instantiate a ' +
      'controller on the window. This behavior is deprecated in Angular ' +
      '1.3.0') {
    return true; // An ngHint bug, see http://git.io/S3yySQ
  }

  var module = /^Module "(\w*)" was created but never loaded\.$/.exec(
      message);
  if (module != null) {
    module = module[1];
    if (ngHintNames[module] != null) {
      return true; // An ngHint module
    }
    if ((module == 'protractorBaseModule_') || (module ==
        'protractorNgHintCaptureModule_')) {
      return true; // A protractor module
    }
  }

  return false;
}

/**
 * Checks the information which has been stored by the ngHint plugin and
 * generates passed/failed tests and/or console output
 *
 * @param {Object} config The configuration file for the ngHint plugin
 * @return {q.Promise} A promise which resolves to the results of any passed or
 *    failed tests
 * @public
 */
function teardown(config) {
  // Get logged data
  return browser.executeScript_(function() {
    return localStorage.getItem('ngHintLog_protractor') || '{}';
  }, 'get ngHintLog').then(function(ngHintLog) {
    ngHintLog = JSON.parse(ngHintLog);

    // Get a list of all the modules we tested against
    var modulesUsed = _.union.apply(_, [_.values(ngHintNames)].concat(
        _.values(ngHintLog).map(Object.keys)));

    // Check log
    var testOut = {failedCount: 0, specResults: []};
    for (url in ngHintLog) {
      if (!isExcluded(url, config)) {
        for (var i = 0; i < modulesUsed.length; i++) {
          // Add new test to output
          var assertions = [];
          testOut.specResults.push({
            description: 'Angular Hint Test: ' + modulesUsed[i] + ' (' + url +
                ')',
            assertions: assertions,
            duration: 1
          });

          // Fill in the test details
          var messages = ngHintLog[url][modulesUsed[i]];
          if (messages) {
            for (var message in messages) {
              if (!isMessageToIgnore(message)) {
                assertions.push({
                  passed: false,
                  errorMsg: messages[message] + ' -- ' + message,
                  stackTrace: ''
                });
              }
            }
          }

          if (assertions.length == 0) {
            // Pass
            assertions.push({
              passed: true,
              errorMsg: '',
              stackTrace: ''
            });
          } else {
            // Fail
            testOut.failedCount++;
          }
        }
      }
    }

    // Return
    if (config.asTests == false) {
      for (var i = 0; i < testOut.specResults.length; i++) {
        for (var j = 0; j < testOut.specResults[i].assertions.length; j++) {
          var assertion = testOut.specResults[i].assertions[j];
          if (!assertion.passed) {
            console.log(assertion.errorMsg);
          }
        }
      }
    } else if ((testOut.failedCount > 0) || (testOut.specResults.length > 0)) {
      return testOut;
    }
  });
}

// Export
exports.setup = setup;
exports.teardown = teardown;
