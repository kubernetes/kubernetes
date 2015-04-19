"use strict";
var vows = require('vows')
, assert = require('assert')
, sandbox = require('sandboxed-module')
, fakeConsole = {
  error: function(format, label, message) {
    this.logged = [ format, label, message ];
  }
}
, globals = function(debugValue) {
  return {
    process: {
      env: {
        'NODE_DEBUG': debugValue
      }
    },
    console: fakeConsole
  };
};

vows.describe('../lib/debug').addBatch({
  'when NODE_DEBUG is set to log4js': {
    topic: function() {
      var debug = sandbox.require(
        '../lib/debug', 
        { 'globals': globals('log4js') }
      );

      fakeConsole.logged = [];
      debug('cheese')('biscuits');
      return fakeConsole.logged;
    },
    'it should log to console.error': function(logged) {
      assert.equal(logged[0], 'LOG4JS: (%s) %s');
      assert.equal(logged[1], 'cheese');
      assert.equal(logged[2], 'biscuits');
    }
  },

  'when NODE_DEBUG is set to not log4js': {
    topic: function() {
      var debug = sandbox.require(
        '../lib/debug',
        { globals: globals('other_module') }
      );

      fakeConsole.logged = [];
      debug('cheese')('biscuits');
      return fakeConsole.logged;
    },
    'it should not log to console.error': function(logged) {
      assert.equal(logged.length, 0);
    }
  },

  'when NODE_DEBUG is not set': {
    topic: function() {
      var debug = sandbox.require(
        '../lib/debug',
        { globals: globals(null) }
      );

      fakeConsole.logged = [];
      debug('cheese')('biscuits');
      return fakeConsole.logged;
    },
    'it should not log to console.error': function(logged) {
      assert.equal(logged.length, 0);
    }
  }

}).exportTo(module);
