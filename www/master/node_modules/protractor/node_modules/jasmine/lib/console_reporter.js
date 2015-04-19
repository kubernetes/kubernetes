var path = require('path');

module.exports = exports = ConsoleReporter;

var noopTimer = {
  start: function(){},
  elapsed: function(){ return 0; }
};

function ConsoleReporter(options) {
  var print = options.print,
    showColors = options.showColors || false,
    onComplete = options.onComplete || function() {},
    timer = options.timer || noopTimer,
    jasmineCorePath = options.jasmineCorePath,
    specCount,
    failureCount,
    failedSpecs = [],
    pendingCount,
    ansi = {
      green: '\x1B[32m',
      red: '\x1B[31m',
      yellow: '\x1B[33m',
      none: '\x1B[0m'
    },
    failedSuites = [];

  this.jasmineStarted = function() {
    specCount = 0;
    failureCount = 0;
    pendingCount = 0;
    print('Started');
    printNewline();
    timer.start();
  };

  this.jasmineDone = function() {
    printNewline();
    printNewline();
    if(failedSpecs.length > 0) {
      print('Failures:');
    }
    for (var i = 0; i < failedSpecs.length; i++) {
      specFailureDetails(failedSpecs[i], i + 1);
    }

    if(specCount > 0) {
      printNewline();

      var specCounts = specCount + ' ' + plural('spec', specCount) + ', ' +
        failureCount + ' ' + plural('failure', failureCount);

      if (pendingCount) {
        specCounts += ', ' + pendingCount + ' pending ' + plural('spec', pendingCount);
      }

      print(specCounts);
    } else {
      print('No specs found');
    }

    printNewline();
    var seconds = timer.elapsed() / 1000;
    print('Finished in ' + seconds + ' ' + plural('second', seconds));
    printNewline();

    for(i = 0; i < failedSuites.length; i++) {
      suiteFailureDetails(failedSuites[i]);
    }

    onComplete(failureCount === 0);
  };

  this.specDone = function(result) {
    specCount++;

    if (result.status == 'pending') {
      pendingCount++;
      print(colored('yellow', '*'));
      return;
    }

    if (result.status == 'passed') {
      print(colored('green', '.'));
      return;
    }

    if (result.status == 'failed') {
      failureCount++;
      failedSpecs.push(result);
      print(colored('red', 'F'));
    }
  };

  this.suiteDone = function(result) {
    if (result.failedExpectations && result.failedExpectations.length > 0) {
      failureCount++;
      failedSuites.push(result);
    }
  };

  return this;

  function printNewline() {
    print('\n');
  }

  function colored(color, str) {
    return showColors ? (ansi[color] + str + ansi.none) : str;
  }

  function plural(str, count) {
    return count == 1 ? str : str + 's';
  }

  function repeat(thing, times) {
    var arr = [];
    for (var i = 0; i < times; i++) {
      arr.push(thing);
    }
    return arr;
  }

  function indent(str, spaces) {
    var lines = (str || '').split('\n');
    var newArr = [];
    for (var i = 0; i < lines.length; i++) {
      newArr.push(repeat(' ', spaces).join('') + lines[i]);
    }
    return newArr.join('\n');
  }

  function filterStack(stack) {
    var filteredStack = stack.split('\n').filter(function(stackLine) {
      return stackLine.indexOf(jasmineCorePath) === -1;
    }).join('\n');
    return filteredStack;
  }

  function specFailureDetails(result, failedSpecNumber) {
    printNewline();
    print(failedSpecNumber + ') ');
    print(result.fullName);

    for (var i = 0; i < result.failedExpectations.length; i++) {
      var failedExpectation = result.failedExpectations[i];
      printNewline();
      print(indent('Message:', 2));
      printNewline();
      print(colored('red', indent(failedExpectation.message, 4)));
      printNewline();
      print(indent('Stack:', 2));
      printNewline();
      print(indent(filterStack(failedExpectation.stack), 4));
    }

    printNewline();
  }

  function suiteFailureDetails(result) {
    for (var i = 0; i < result.failedExpectations.length; i++) {
      printNewline();
      print(colored('red', 'An error was thrown in an afterAll'));
      printNewline();
      print(colored('red', 'AfterAll ' + result.failedExpectations[i].message));

    }
    printNewline();
  }
}
