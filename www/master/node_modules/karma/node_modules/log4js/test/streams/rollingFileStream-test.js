"use strict";
var vows = require('vows')
, async = require('async')
, assert = require('assert')
, events = require('events')
, fs = require('fs')
, semver = require('semver')
, streams
, RollingFileStream;

if (semver.satisfies(process.version, '>=0.10.0')) {
  streams = require('stream');
} else {
  streams = require('readable-stream');
}
RollingFileStream = require('../../lib/streams').RollingFileStream;

function remove(filename) {
  try {
    fs.unlinkSync(filename);
  } catch (e) {
    //doesn't really matter if it failed
  }
}

function create(filename) {
  fs.writeFileSync(filename, "test file");
}

vows.describe('RollingFileStream').addBatch({
  'arguments': {
    topic: function() {
      remove(__dirname + "/test-rolling-file-stream");
      return new RollingFileStream("test-rolling-file-stream", 1024, 5);
    },
    'should take a filename, file size (bytes), no. backups,  return Writable': function(stream) {
      assert.instanceOf(stream, streams.Writable);
      assert.equal(stream.filename, "test-rolling-file-stream");
      assert.equal(stream.size, 1024);
      assert.equal(stream.backups, 5);
    },
    'with default settings for the underlying stream': function(stream) {
      assert.equal(stream.theStream.mode, 420);
      assert.equal(stream.theStream.flags, 'a');
      //encoding isn't a property on the underlying stream
      //assert.equal(stream.theStream.encoding, 'utf8');
    }
  },
  'with stream arguments': {
    topic: function() {
      remove(__dirname + '/test-rolling-file-stream');
      return new RollingFileStream(
        'test-rolling-file-stream', 
        1024, 
        5, 
        { mode: parseInt('0666', 8) }
      );
    },
    'should pass them to the underlying stream': function(stream) {
      assert.equal(stream.theStream.mode, parseInt('0666', 8));
    }
  },
  'without size': {
    topic: function() {
      try {
        new RollingFileStream(__dirname + "/test-rolling-file-stream");
      } catch (e) {
        return e;
      }
    },
    'should throw an error': function(err) {
      assert.instanceOf(err, Error);
    }
  },
  'without number of backups': {
    topic: function() {
      remove('test-rolling-file-stream');
      return new RollingFileStream(__dirname + "/test-rolling-file-stream", 1024);
    },
    'should default to 1 backup': function(stream) {
      assert.equal(stream.backups, 1);
    }
  },
  'writing less than the file size': {
    topic: function() {
      remove(__dirname + "/test-rolling-file-stream-write-less");
      var that = this
      , stream = new RollingFileStream(
        __dirname + "/test-rolling-file-stream-write-less", 
        100
      );
      stream.write("cheese", "utf8", function() {
        stream.end();
        fs.readFile(__dirname + "/test-rolling-file-stream-write-less", "utf8", that.callback);
      });
    },
    'should write to the file': function(contents) {
      assert.equal(contents, "cheese");
    },
    'the number of files': {
      topic: function() {
        fs.readdir(__dirname, this.callback);
      },
      'should be one': function(files) {
        assert.equal(
          files.filter(
            function(file) { 
              return file.indexOf('test-rolling-file-stream-write-less') > -1; 
            }
          ).length, 
          1
        );
      }
    }
  },
  'writing more than the file size': {
    topic: function() {
      remove(__dirname + "/test-rolling-file-stream-write-more");
      remove(__dirname + "/test-rolling-file-stream-write-more.1");
      var that = this
      , stream = new RollingFileStream(
        __dirname + "/test-rolling-file-stream-write-more", 
        45
      );
      async.each(
        [0, 1, 2, 3, 4, 5, 6], 
        function(i, cb) {
          stream.write(i +".cheese\n", "utf8", cb);
        }, 
        function() {
          stream.end();
          that.callback();
        }
      );
    },
    'the number of files': {
      topic: function() {
        fs.readdir(__dirname, this.callback);
      },
      'should be two': function(files) {
        assert.equal(files.filter(
          function(file) { 
            return file.indexOf('test-rolling-file-stream-write-more') > -1; 
          }
        ).length, 2);
      }
    },
    'the first file': {
      topic: function() {
        fs.readFile(__dirname + "/test-rolling-file-stream-write-more", "utf8", this.callback);
      },
      'should contain the last two log messages': function(contents) {
        assert.equal(contents, '5.cheese\n6.cheese\n');
      }
    },
    'the second file': {
      topic: function() {
        fs.readFile(__dirname + '/test-rolling-file-stream-write-more.1', "utf8", this.callback);
      },
      'should contain the first five log messages': function(contents) {
        assert.equal(contents, '0.cheese\n1.cheese\n2.cheese\n3.cheese\n4.cheese\n');
      }
    }
  },
  'when many files already exist': {
    topic: function() {
      remove(__dirname + '/test-rolling-stream-with-existing-files.11');
      remove(__dirname + '/test-rolling-stream-with-existing-files.20');
      remove(__dirname + '/test-rolling-stream-with-existing-files.-1');
      remove(__dirname + '/test-rolling-stream-with-existing-files.1.1');
      remove(__dirname + '/test-rolling-stream-with-existing-files.1');
      

      create(__dirname + '/test-rolling-stream-with-existing-files.11');
      create(__dirname + '/test-rolling-stream-with-existing-files.20');
      create(__dirname + '/test-rolling-stream-with-existing-files.-1');
      create(__dirname + '/test-rolling-stream-with-existing-files.1.1');
      create(__dirname + '/test-rolling-stream-with-existing-files.1');

      var that = this
      , stream = new RollingFileStream(
        __dirname + "/test-rolling-stream-with-existing-files", 
        45,
        5
      );
      async.each(
        [0, 1, 2, 3, 4, 5, 6], 
        function(i, cb) {
          stream.write(i +".cheese\n", "utf8", cb);
        }, 
        function() {
          stream.end();
          that.callback();
        }
      );
    },
    'the files': {
      topic: function() {
        fs.readdir(__dirname, this.callback);
      },
      'should be rolled': function(files) {
        assert.include(files, 'test-rolling-stream-with-existing-files');
        assert.include(files, 'test-rolling-stream-with-existing-files.1');
        assert.include(files, 'test-rolling-stream-with-existing-files.2');
        assert.include(files, 'test-rolling-stream-with-existing-files.11');
        assert.include(files, 'test-rolling-stream-with-existing-files.20');
      }
    }
  }  
}).exportTo(module);
