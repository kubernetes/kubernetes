var util = require('util');
var path = require('path');

function noop() {}

// Put jasmine in the global context, this is somewhat like running in a
// browser where every file will have access to `jasmine`.
var requireJasmine = require('./jasmine-1.3.1.js');
for (var key in requireJasmine) {
  global[key] = requireJasmine[key];
}

// Overwriting it allows us to handle custom async specs.
global.it = function(desc, func, timeout) {
    return jasmine.getEnv().it(desc, func, timeout);
};
global.beforeEach = function(func, timeout) {
    return jasmine.getEnv().beforeEach(func, timeout);
};
global.afterEach = function(func, timeout) {
    return jasmine.getEnv().afterEach(func, timeout);
};

// Allow async tests by passing in a 'done' function.
require('./async-callback');

// For the terminal reporters.
var nodeReporters = require('./reporter');
jasmine.TerminalVerboseReporter = nodeReporters.TerminalVerboseReporter;
jasmine.TerminalReporter = nodeReporters.TerminalReporter;

function removeJasmineFrames(text) {
  if (!text) {
    return text;
  }
  var jasmineFilename = __dirname + '/jasmine-1.3.1.js';
  var lines = [];
  text.split(/\n/).forEach(function(line){
    if (line.indexOf(jasmineFilename) == -1) {
      lines.push(line);
    }
  });
  return lines.join('\n');
}

var specFiles = [];

/**
 * Add a spec file to the list to be executed. Specs should be relative
 * to the current working dir of the process or absolute.
 * @param {string|Array.<string>} specs
 */
exports.addSpecs = function(specs) {
  if (typeof specs === 'string') {
    specFiles.push(specs);
  } else if (specs.length) {
    for (var i = 0; i < specs.length; ++i) {
      specFiles.push(specs[i]);
    }
  }
};

/**
 * Alias for jasmine.getEnv().addReporter
 */
exports.addReporter = function (reporter) {
  jasmine.getEnv().addReporter(reporter);
};

/**
 * Execute the loaded specs. Optional options object described below.
 * @param {Object} options
 */
exports.executeSpecs = function(options) {
  options = options || {};
  // An array of filenames, either absolute or relative to current working dir.
  // These will be executed, as well as any tests added with addSpecs()
  var specs = options['specs'] || [];
  // A function to call on completion. function(runner, log)
  var done = options['onComplete'];
  // If true, display spec names
  var isVerbose = options['isVerbose'];
  // If true, output nothing to the terminal. This overrides other output options.
  var silent = options['silent'];
  // If true, print colors to the terminal
  var showColors = options['showColors'];
  // If true, include stack traces in failures
  var includeStackTrace = options['includeStackTrace'];
  // Time to wait in milliseconds before a test automatically fails
  var defaultTimeoutInterval = options['defaultTimeoutInterval'] || 5000;
  // Jasmine environment to use.
  var jasmineEnv = options['jasmineEnv'] || jasmine.getEnv();
  // Overrides the print function of the terminal reporters
  var print = options['print'] || process.stdout.write.bind(process.stdout);
  // Overrides the stack trace filter
  var stackFilter = options['stackFilter'] || removeJasmineFrames;
  // Show timing information on failures
  var showTiming = options['showTiming'];
  // Print failures in real time.
  var realtimeFailure = options['realtimeFailure'];

  if (silent) {
    print = noop;
  }

  var originalGetEnv = jasmine.getEnv;
  jasmine.getEnv = function() {
    return jasmineEnv;
  }

  jasmineEnv.addReporter(new jasmine.TerminalReporter({
    print:       print,
    color: showColors,
    includeStackTrace: includeStackTrace,
    isVerbose: isVerbose,
    onComplete:  done,
    stackFilter: stackFilter,
    showTiming: showTiming,
    realtimeFailure: realtimeFailure}));

  jasmineEnv.defaultTimeoutInterval = defaultTimeoutInterval;

  specFiles = specFiles.concat(specs);

  for (var i = 0, len = specFiles.length; i < len; ++i) {
    var filename = specFiles[i];
    // Catch exceptions in loading the spec files, and make them jasmine test
    // failures.
    try {
      require(path.resolve(process.cwd(), filename));
    } catch (e) {
      // Generate a synthetic suite with a failure spec, so that the failure is
      // reported with other results.
      jasmineEnv.describe('Exception loading: ' + filename, function() {
        jasmineEnv.it('Error', function() { throw e; });
      });
    }
  }

  jasmineEnv.execute();

  jasmine.getEnv = originalGetEnv;

};
