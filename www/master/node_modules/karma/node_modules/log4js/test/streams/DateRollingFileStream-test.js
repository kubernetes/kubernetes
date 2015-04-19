"use strict";
var vows = require('vows')
, assert = require('assert')
, fs = require('fs')
, semver = require('semver')
, streams
, DateRollingFileStream
, testTime = new Date(2012, 8, 12, 10, 37, 11);

if (semver.satisfies(process.version, '>=0.10.0')) {
  streams = require('stream');
} else {
  streams = require('readable-stream');
}
DateRollingFileStream = require('../../lib/streams').DateRollingFileStream;

function cleanUp(filename) {
  return function() {
    fs.unlink(filename);
  };
}

function now() {
  return testTime.getTime();
}

vows.describe('DateRollingFileStream').addBatch({
  'arguments': {
    topic: new DateRollingFileStream(
      __dirname + '/test-date-rolling-file-stream-1', 
      'yyyy-mm-dd.hh'
    ),
    teardown: cleanUp(__dirname + '/test-date-rolling-file-stream-1'),
    
    'should take a filename and a pattern and return a WritableStream': function(stream) {
      assert.equal(stream.filename, __dirname + '/test-date-rolling-file-stream-1');
      assert.equal(stream.pattern, 'yyyy-mm-dd.hh');
      assert.instanceOf(stream, streams.Writable);
    },
    'with default settings for the underlying stream': function(stream) {
      assert.equal(stream.theStream.mode, 420);
      assert.equal(stream.theStream.flags, 'a');
      //encoding is not available on the underlying stream
      //assert.equal(stream.encoding, 'utf8');
    }
  },
  
  'default arguments': {
    topic: new DateRollingFileStream(__dirname + '/test-date-rolling-file-stream-2'),
    teardown: cleanUp(__dirname + '/test-date-rolling-file-stream-2'),
    
    'pattern should be .yyyy-MM-dd': function(stream) {
      assert.equal(stream.pattern, '.yyyy-MM-dd');
    }
  },

  'with stream arguments': {
    topic: new DateRollingFileStream(
      __dirname + '/test-date-rolling-file-stream-3', 
      'yyyy-MM-dd', 
      { mode: parseInt('0666', 8) }
    ),
    teardown: cleanUp(__dirname + '/test-date-rolling-file-stream-3'),
    
    'should pass them to the underlying stream': function(stream) {
      assert.equal(stream.theStream.mode, parseInt('0666', 8));
    }
  },

  'with stream arguments but no pattern': {
    topic: new DateRollingFileStream(
      __dirname + '/test-date-rolling-file-stream-4', 
      { mode: parseInt('0666', 8) }
    ),
    teardown: cleanUp(__dirname + '/test-date-rolling-file-stream-4'),
    
    'should pass them to the underlying stream': function(stream) {
      assert.equal(stream.theStream.mode, parseInt('0666', 8));
    },
    'should use default pattern': function(stream) {
      assert.equal(stream.pattern, '.yyyy-MM-dd');
    }
  },

  'with a pattern of .yyyy-MM-dd': {
    topic: function() {
      var that = this,
      stream = new DateRollingFileStream(
        __dirname + '/test-date-rolling-file-stream-5', '.yyyy-MM-dd', 
        null, 
        now
      );
      stream.write("First message\n", 'utf8', function() {
        that.callback(null, stream);
      });
    },
    teardown: cleanUp(__dirname + '/test-date-rolling-file-stream-5'),
    
    'should create a file with the base name': {
      topic: function(stream) {
        fs.readFile(__dirname + '/test-date-rolling-file-stream-5', this.callback);
      },
      'file should contain first message': function(result) {
        assert.equal(result.toString(), "First message\n");
      }
    },

    'when the day changes': {
      topic: function(stream) {
        testTime = new Date(2012, 8, 13, 0, 10, 12);
        stream.write("Second message\n", 'utf8', this.callback);
      },
      teardown: cleanUp(__dirname + '/test-date-rolling-file-stream-5.2012-09-12'),

      
      'the number of files': {
        topic: function() {
          fs.readdir(__dirname, this.callback);
        },
        'should be two': function(files) {
          assert.equal(
            files.filter(
              function(file) { 
                return file.indexOf('test-date-rolling-file-stream-5') > -1; 
              }
            ).length, 
            2
          );
        }
      },
      
      'the file without a date': {
        topic: function() {
          fs.readFile(__dirname + '/test-date-rolling-file-stream-5', this.callback);
        },
        'should contain the second message': function(contents) {
          assert.equal(contents.toString(), "Second message\n");
        }
      },
      
      'the file with the date': {
        topic: function() {
          fs.readFile(__dirname + '/test-date-rolling-file-stream-5.2012-09-12', this.callback);
        },
        'should contain the first message': function(contents) {
          assert.equal(contents.toString(), "First message\n");
        }
      }
    }
  },
  
  'with alwaysIncludePattern': {
    topic: function() {
      var that = this,
      testTime = new Date(2012, 8, 12, 0, 10, 12),
      stream = new DateRollingFileStream(
        __dirname + '/test-date-rolling-file-stream-pattern', 
        '.yyyy-MM-dd', 
        {alwaysIncludePattern: true}, 
        now
      );
      stream.write("First message\n", 'utf8', function() {
        that.callback(null, stream);
      });
    },
    teardown: cleanUp(__dirname + '/test-date-rolling-file-stream-pattern.2012-09-12'),
    
    'should create a file with the pattern set': {
      topic: function(stream) {
        fs.readFile(__dirname + '/test-date-rolling-file-stream-pattern.2012-09-12', this.callback);
      },
      'file should contain first message': function(result) {
        assert.equal(result.toString(), "First message\n");
      }
    },
    
    'when the day changes': {
      topic: function(stream) {
        testTime = new Date(2012, 8, 13, 0, 10, 12);
        stream.write("Second message\n", 'utf8', this.callback);
      },
      teardown: cleanUp(__dirname + '/test-date-rolling-file-stream-pattern.2012-09-13'),
      
      
      'the number of files': {
        topic: function() {
          fs.readdir(__dirname, this.callback);
        },
        'should be two': function(files) {
          assert.equal(
            files.filter(
              function(file) { 
                return file.indexOf('test-date-rolling-file-stream-pattern') > -1; 
              }
            ).length, 
            2
          );
        }
      },
      
      'the file with the later date': {
        topic: function() {
          fs.readFile(
            __dirname + '/test-date-rolling-file-stream-pattern.2012-09-13', 
            this.callback
          );
        },
        'should contain the second message': function(contents) {
          assert.equal(contents.toString(), "Second message\n");
        }
      },
      
      'the file with the date': {
        topic: function() {
          fs.readFile(
            __dirname + '/test-date-rolling-file-stream-pattern.2012-09-12', 
            this.callback
          );
        },
        'should contain the first message': function(contents) {
          assert.equal(contents.toString(), "First message\n");
        }
      }
    }
  }

}).exportTo(module);
