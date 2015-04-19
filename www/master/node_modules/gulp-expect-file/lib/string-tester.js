'use strict';

var async = require('async');
var wrapAssertion = require('./util').wrapAssertion;

module.exports = StringTester;

function StringTester(expectation) {
  this.tester = this.parseExpectation(expectation);
}

StringTester.prototype.test = function (target, callback) {
  this.tester(target, callback);
};

StringTester.prototype.parseExpectation = function (expectation) {
  if (expectation instanceof Array) {
    var expectations = expectation.map(this.parseExpectation.bind(this));
    return function (target, callback) {
      async.applyEachSeries(expectations, target, callback);
    };
  }

  if (typeof expectation === 'function') {
    return wrapAssertion(expectation);
  }

  if (typeof expectation === 'string') {
    return wrapAssertion(
      function (target) { return target.indexOf(expectation) >= 0; },
      'not containing ' + JSON.stringify(expectation)
    );
  }
  if (expectation instanceof RegExp) {
    return wrapAssertion(
      function (target) { return expectation.test(target); },
      'not matching ' + expectation.toString()
    );
  }

  throw new TypeError('Unknown expectation type');
};
