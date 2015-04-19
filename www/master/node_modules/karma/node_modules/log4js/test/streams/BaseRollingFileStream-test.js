"use strict";
var vows = require('vows')
, assert = require('assert')
, fs = require('fs')
, sandbox = require('sandboxed-module');

vows.describe('../../lib/streams/BaseRollingFileStream').addBatch({
  'when node version < 0.10.0': {
    topic: function() {
      var streamLib = sandbox.load(
        '../../lib/streams/BaseRollingFileStream',
        {
          globals: {
            process: {
              version: '0.8.11'
            }
          },
          requires: {
            'readable-stream': {
              Writable: function() {}
            }
          }
        }
      );
      return streamLib.required;
    },
    'it should use readable-stream to maintain compatibility': function(required) {
      assert.ok(required['readable-stream']);
      assert.ok(!required.stream);
    }
  },

  'when node version > 0.10.0': {
    topic: function() {
      var streamLib = sandbox.load(
        '../../lib/streams/BaseRollingFileStream',
        {
          globals: {
            process: {
              version: '0.10.1'
            }
          },
          requires: {
            'stream': {
              Writable: function() {}
            }
          }
        }
      );
      return streamLib.required;
    },
    'it should use the core stream module': function(required) {
      assert.ok(required.stream);
      assert.ok(!required['readable-stream']);
    }
  },

  'when no filename is passed': {
    topic: require('../../lib/streams/BaseRollingFileStream'),
    'it should throw an error': function(BaseRollingFileStream) {
      try {
        new BaseRollingFileStream();
        assert.fail('should not get here');
      } catch (e) {
        assert.ok(e);
      }
    }
  },

  'default behaviour': {
    topic: function() {
      var BaseRollingFileStream = require('../../lib/streams/BaseRollingFileStream')
      , stream = new BaseRollingFileStream('basetest.log');
      return stream;
    },
    teardown: function() {
      try {
        fs.unlink('basetest.log');
      } catch (e) {
        console.error("could not remove basetest.log", e);
      }
    },
    'it should not want to roll': function(stream) {
      assert.isFalse(stream.shouldRoll());
    },
    'it should not roll': function(stream) {
      var cbCalled = false;
      //just calls the callback straight away, no async calls
      stream.roll('basetest.log', function() { cbCalled = true; });
      assert.isTrue(cbCalled);
    }
  }
}).exportTo(module);
