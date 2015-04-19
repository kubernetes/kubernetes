var q = require('q'),
    path = require('path');

var STACK_SUBSTRINGS_TO_FILTER = [
  'node_modules/minijasminenode/lib/',
  'node_modules/selenium-webdriver',
  'at Module.',
  'at Object.Module.',
  'at Function.Module',
  '(timers.js:',
  'jasminewd/index.js',
  'protractor/lib/'
];


/**
 * Utility function that filters a stack trace to be more readable. It removes
 * Jasmine test frames and webdriver promise resolution.
 * @param {string} text Original stack trace.
 * @return {string}
 */
exports.filterStackTrace = function(text) {
  if (!text) {
    return text;
  }
  var lines = text.split(/\n/).filter(function(line) {
    for (var i = 0; i < STACK_SUBSTRINGS_TO_FILTER.length; ++i) {
      if (line.indexOf(STACK_SUBSTRINGS_TO_FILTER[i]) !== -1) {
        return false;
      }
    }
    return true;
  });
  return lines.join('\n');
};

/**
 * Internal helper for abstraction of polymorphic filenameOrFn properties.
 * @param {object} filenameOrFn The filename or function that we will execute.
 * @param {Array.<object>}} args The args to pass into filenameOrFn.
 * @return {q.Promise} A promise that will resolve when filenameOrFn completes.
 */
exports.runFilenameOrFn_ = function(configDir, filenameOrFn, args) {
  return q.promise(function(resolve) {
    if (filenameOrFn &&
        !(typeof filenameOrFn === 'string' || typeof filenameOrFn === 'function')) {
      throw 'filenameOrFn must be a string or function';
    }

    if (typeof filenameOrFn === 'string') {
      filenameOrFn = require(path.resolve(configDir, filenameOrFn));
    }
    if (typeof filenameOrFn === 'function') {
      var results = q.when(filenameOrFn.apply(null, args), null, function(err) {
        err.stack = exports.filterStackTrace(err.stack);
        throw err;
      });
      resolve(results);
    } else {
      resolve();
    }
  });
};

/**
 * Joins two logs of test results, each following the format of <framework>.run
 * @param {object} log1
 * @param {object} log2
 * @return {object} The joined log
 */
exports.joinTestLogs = function(log1, log2) {
  return {failedCount: log1.failedCount + log2.failedCount,
          specResults: (log1.specResults || []).concat(log2.specResults || [])
  };
};
