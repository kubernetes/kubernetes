var util = require('util');

//
// Helpers
//
function noop() {}

ANSI_COLORS = {
  pass:    function() { return '\033[32m'; }, // Green
  fail:    function() { return '\033[31m'; }, // Red
  neutral: function() { return '\033[0m';  }  // Normal
},

NO_COLORS = {
  pass:    function() { return ''; },
  fail:    function() { return ''; },
  neutral: function() { return ''; }
},

TerminalReporter = function(config) {
  // Options from the configuration.
  this.print_ = config.print || console.log;
  this.color_ = config.color ? ANSI_COLORS : NO_COLORS;
  this.includeStackTrace_ = config.includeStackTrace === false ? false : true;
  this.stackFilter = config.stackFilter || false;
  this.showTiming_ = config.showTiming || false;
  this.callback_ = config.onComplete || false;
  this.isVerbose_ = config.isVerbose || false;
  this.realtimeFailure_ = config.realtimeFailure || false;

  this.visitedSuites_ = {};

  /** @type {boolean} */
  this.started_ = false;
  /** @type {boolean} */
  this.finished_ = false;
  /** @type {Date} */
  this.startedAt = null;
  /** @type {number} */
  this.numFailures_ = 0;

  /** @type {{description: string, message: string, stackTrace: string, finishTime: string}} */
  this.failures_ = [];
}


TerminalReporter.prototype = {
  reportRunnerStarting: function(runner) {
    this.started_ = true;
    this.startedAt = new Date();
  },

  // This is heavily influenced by Jasmine's Html/Trivial Reporter
  reportRunnerResults: function(runner) {
    this.reportFailures_();

    var results = runner.results();
    var resultColor = (results.failedCount > 0) ? this.color_.fail() : this.color_.pass();

    var specs = runner.specs();
    var specCount = specs.length;

    var message = "\n\nFinished in " +
        ((new Date().getTime() - this.startedAt.getTime()) / 1000) + " seconds";
    this.printLine_(message);

    this.printLine_(this.stringWithColor_(this.printRunnerResults_(runner), resultColor));

    this.finished_ = true;
    if(this.callback_) { this.callback_(runner); }
  },

  reportFailures_: function() {
    if (this.failures_.length === 0) {
      return;
    }

    this.printLine_('\n');

    this.print_('Failures:');

    for (var i = 0; i < this.failures_.length; i++) {
      this.printFailure_(this.failures_[i], i);
    }
  },

  printFailure_: function(failure, index) {
    this.printLine_('\n');
    this.printLine_('  ' + (index + 1) + ') ' + failure.description);
    if (this.showTiming_) {
      this.printLine_('  at ' + failure.finishTime / 1000 + 's' +
          ' [' + failure.finishedAt.toUTCString() + ']');
    }
    this.printLine_('   Message:');
    this.printLine_('     ' + this.stringWithColor_(failure.message, this.color_.fail()));
    if (this.includeStackTrace_) {
      this.printLine_('   Stacktrace:');
      var stackTrace = failure.stackTrace;
      if (this.stackFilter) {
        stackTrace = this.stackFilter(stackTrace);
      }
      this.print_('     ' + stackTrace);
    }
  },

  reportSuiteResults: function(suite) {
    // Not used in this context
  },

  reportSpecResults: function(spec) {
    spec.finishedAt = new Date();
    spec.finishTime = spec.finishedAt.getTime() - this.startedAt.getTime();
    var result = spec.results();
    var msg = '';
    var passed = result.passed();

    if (this.isVerbose_) {
      // There is no 'suiteStarted' event for Jasmine 1.3.X, so we re-calculate
      // the active suites every time.
      var indent = 0;
      var suite = spec.suite;
      var suiteChain = [];
      while (suite) {
        suiteChain.push(suite);
        suite = suite.parentSuite;
      }
      for (var i = 0; i < suiteChain.length; i++) {
        var suite = suiteChain[suiteChain.length - i - 1];
        if (!this.visitedSuites_[suite.id]) {
          this.visitedSuites_[suite.id] = true;
          this.printLine_(this.indentMessage_(suite.description, i));
        }
        indent++;
      }
      if (passed) {
        this.printLine_(this.stringWithColor_(
            this.indentMessage_(spec.description + ' - pass', indent), this.color_.pass()));
      } else {
        this.printLine_(this.stringWithColor_(
            this.indentMessage_(spec.description + ' - fail', indent), this.color_.fail()));
        this.addFailure_(spec);
      }
    } else {
      if (passed) {
        this.print_(this.stringWithColor_('.', this.color_.pass()));
      } else {
        this.print_(this.stringWithColor_('F', this.color_.fail()));
        this.addFailure_(spec);
      }
    }
  },

  addFailure_: function(spec) {
    var result = spec.results();
    var failureItem = null;

    var items_length = result.items_.length;
    for (var i = 0; i < items_length; i++) {
      if (result.items_[i].passed_ === false) {
        failureItem = result.items_[i];

        var failure = {
          description: spec.suite.getFullName() + " " + spec.description,
          message: failureItem.message,
          stackTrace: failureItem.trace.stack,
          finishTime: spec.finishTime,
          finishedAt: spec.finishedAt
        }

        if (this.realtimeFailure_) {
          this.printFailure_(failure, this.numFailures_++);
        } else {
          this.failures_.push(failure);
        }
      }
    }
  },

  printRunnerResults_: function(runner){
    var results = runner.results();
    var specs = runner.specs();
    var msg = '';
    msg += specs.length + ' test' + ((specs.length === 1) ? '' : 's') + ', ';
    msg += results.totalCount + ' assertion' + ((results.totalCount === 1) ? '' : 's') + ', ';
    msg += results.failedCount + ' failure' + ((results.failedCount === 1) ? '' : 's') + '\n';
    return msg;
  },

    // Helper Methods //
  stringWithColor_: function(stringValue, color) {
    return (color || this.color_.neutral()) + stringValue + this.color_.neutral();
  },

  printLine_: function(stringValue) {
    this.print_(stringValue);
    this.print_('\n');
  },

  indentMessage_: function(message, indentCount) {
    var _indent = '';
    for (var i = 0; i < indentCount; i++) {
      _indent += '  ';
    }
    return (_indent + message);
  }
};

exports.TerminalReporter = TerminalReporter;
