#!/usr/bin/env node
require('../global');

var path = require('path');

var failed = false;

//
// Lint
//
JSHINT_BIN = './node_modules/jshint/bin/jshint';
cd(__dirname + '/..');

if (!test('-f', JSHINT_BIN)) {
  echo('JSHint not found. Run `npm install` in the root dir first.');
  exit(1);
}

if (exec(JSHINT_BIN + ' *.js test/*.js').code !== 0) {
  failed = true;
  echo('*** JSHINT FAILED! (return code != 0)');
  echo();
} else {
  echo('All JSHint tests passed');
  echo();
}

//
// Unit tests
//
cd(__dirname + '/../test');
ls('*.js').forEach(function(file) {
  echo('Running test:', file);
  if (exec('node ' + file).code !== 123) { // 123 avoids false positives (e.g. premature exit)
    failed = true;
    echo('*** TEST FAILED! (missing exit code "123")');
    echo();
  }
});

if (failed) {
  echo();
  echo('*******************************************************');
  echo('WARNING: Some tests did not pass!');
  echo('*******************************************************');
  exit(1);
} else {
  echo();
  echo('All tests passed.');
}
