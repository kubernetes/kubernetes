'use strict';

var StringTester = require('./string-tester');
var StreamTester = require('./stream-tester');
var ExpectationError = require('./errors').ExpectationError;
var wrapAssertion = require('./util').wrapAssertion;
var Minimatch = require('minimatch').Minimatch;
var fs = require('fs');
var async = require('async');
var path = require('path');
var xtend = require('xtend');

module.exports = FileTester;

function FileTester(expectation, options) {
  this.rules = parseExpectation(expectation);
  this.options = xtend({ checkRealFile: false }, options);
}

function parseExpectation(expectation) {
  if (expectation instanceof Array) {
    var nested = expectation.map(parseExpectation);
    return Array.prototype.concat.apply([], nested);
  }
  switch (typeof expectation) {
    case 'string':
      return [new Rule(expectation)];
    case 'function':
      return [new Rule(null, expectation)];
    case 'object':
      return Object.keys(expectation).map(function (path) {
        return new Rule(path, expectation[path] === true ? null : expectation[path]);
      });
    default:
      throw new TypeError('Unknown expectation type');
  }
}

function checkFileExists(file, callback) {
  fs.exists(file.path, function (exists) {
    if (exists) {
      callback();
    } else {
      callback(new ExpectationError('not on filesystem'));
    }
  });
}

FileTester.prototype.test = function (file, callback) {
  var _this = this;
  var matchedAny = false;
  async.eachSeries(this.rules, function (rule, next) {
    if (!rule.matchFilePath(file.relative)) {
      return next(null);
    }
    matchedAny = true;
    rule.testFile(file, next);
  }, function (err) {
    if (err) {
      return callback(err);
    }
    if (!matchedAny) {
      return callback(new ExpectationError('unexpected'));
    }

    if (_this.options.checkRealFile) {
      checkFileExists(file, callback);
    } else {
      callback();
    }
  });
};

FileTester.prototype.getUnusedRules = function () {
  return this.rules.filter(function (rule) { return !rule.isUsed() });
};


function Rule(path, tester) {
  this.path = path;
  this.minimatch = path ? new Minimatch(path) : null;
  this.tester = tester ? Rule.wrapTester(tester) : null;
  this.used = false;
}

Rule.prototype.matchFilePath = function (path) {
  if (this.minimatch) {
    return this.minimatch.match(path);
  } else {
    return true;
  }
};

Rule.prototype.testFile = function (file, callback) {
  this.used = true;
  if (this.tester) {
    this.tester(file, callback);
  } else {
    callback(null);
  }
};

Rule.prototype.isUsed = function () {
  return this.used;
};

Rule.prototype.toString = function () {
  return this.path || '(custom)';
};

Rule.wrapTester = function (tester) {
  if (typeof tester === 'function') {
    return wrapAssertion(tester);
  }

  var stringTester = (tester instanceof StringTester) ? tester : new StringTester(tester);
  var streamTester = new StreamTester(stringTester);
  return function (file, callback) {
    if (file.isNull()) {
      callback(new ExpectationError('not read'));
    } else if (file.isStream()) {
      streamTester.test(file.contents, callback);
    } else if (file.isBuffer()) {
      stringTester.test(file.contents.toString(), callback);
    }
  };
};
