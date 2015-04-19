"use strict";
var vows = require('vows')
, fs = require('fs')
, path = require('path')
, sandbox = require('sandboxed-module')
, log4js = require('../lib/log4js')
, assert = require('assert')
, zlib = require('zlib')
, EOL = require('os').EOL || '\n';

log4js.clearAppenders();

function remove(filename) {
  try {
    fs.unlinkSync(filename);
  } catch (e) {
    //doesn't really matter if it failed
  }
}

vows.describe('log4js fileAppender').addBatch({
  'adding multiple fileAppenders': {
    topic: function () {
      var listenersCount = process.listeners('exit').length
      , logger = log4js.getLogger('default-settings')
      , count = 5, logfile;
      
      while (count--) {
        logfile = path.join(__dirname, '/fa-default-test' + count + '.log');
        log4js.addAppender(require('../lib/appenders/file').appender(logfile), 'default-settings');
      }
      
      return listenersCount;
    },
    
    'does not add more than one `exit` listeners': function (initialCount) {
      assert.ok(process.listeners('exit').length <= initialCount + 1);
    }
  },

  'exit listener': {
    topic: function() {
      var exitListener
      , openedFiles = []
      , fileAppender = sandbox.require(
        '../lib/appenders/file',
        {
          globals: {
            process: {
              on: function(evt, listener) {
                exitListener = listener;
              }
            }
          },
          requires: {
            '../streams': {
              RollingFileStream: function(filename) {
                openedFiles.push(filename);
                
                this.end = function() {
                  openedFiles.shift();
                };

                this.on = function() {};
              }
            }
          }   
        }
      );
      for (var i=0; i < 5; i += 1) {
        fileAppender.appender('test' + i, null, 100);
      }
      assert.isNotEmpty(openedFiles);
      exitListener();
      return openedFiles;
    },
    'should close all open files': function(openedFiles) {
      assert.isEmpty(openedFiles);
    }
  },
  
  'with default fileAppender settings': {
    topic: function() {
      var that = this
      , testFile = path.join(__dirname, '/fa-default-test.log')
      , logger = log4js.getLogger('default-settings');
      remove(testFile);

      log4js.clearAppenders();
      log4js.addAppender(require('../lib/appenders/file').appender(testFile), 'default-settings');
      
      logger.info("This should be in the file.");
      
      setTimeout(function() {
        fs.readFile(testFile, "utf8", that.callback);
      }, 100);
    },
    'should write log messages to the file': function (err, fileContents) {
      assert.include(fileContents, "This should be in the file." + EOL);
    },
    'log messages should be in the basic layout format': function(err, fileContents) {
      assert.match(
        fileContents, 
          /\[\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{3}\] \[INFO\] default-settings - /
      );
    }
  },
  'fileAppender subcategories': {
    topic: function() {
      var that = this;

      log4js.clearAppenders();

      function addAppender(cat) {
        var testFile = path.join(__dirname, '/fa-subcategories-test-'+cat.join('-').replace(/\./g, "_")+'.log');
        remove(testFile);
        log4js.addAppender(require('../lib/appenders/file').appender(testFile), cat);
        return testFile;
      }

      var file_sub1 = addAppender([ 'sub1']);
      
      var file_sub1_sub12$sub1_sub13 = addAppender([ 'sub1.sub12', 'sub1.sub13' ]);
      
      var file_sub1_sub12 = addAppender([ 'sub1.sub12' ]);

      
      var logger_sub1_sub12_sub123 = log4js.getLogger('sub1.sub12.sub123');
      
      var logger_sub1_sub13_sub133 = log4js.getLogger('sub1.sub13.sub133');

      var logger_sub1_sub14 = log4js.getLogger('sub1.sub14');

      var logger_sub2 = log4js.getLogger('sub2');
      

      logger_sub1_sub12_sub123.info('sub1_sub12_sub123');
      
      logger_sub1_sub13_sub133.info('sub1_sub13_sub133');

      logger_sub1_sub14.info('sub1_sub14');

      logger_sub2.info('sub2');
           
      
      setTimeout(function() {
        that.callback(null, {
          file_sub1: fs.readFileSync(file_sub1).toString(),
          file_sub1_sub12$sub1_sub13: fs.readFileSync(file_sub1_sub12$sub1_sub13).toString(),
          file_sub1_sub12: fs.readFileSync(file_sub1_sub12).toString()
        });        
      }, 3000);
    },
    'check file contents': function (err, fileContents) {

      // everything but category 'sub2'
      assert.match(fileContents.file_sub1, /^(\[\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{3}\] \[INFO\] (sub1.sub12.sub123 - sub1_sub12_sub123|sub1.sub13.sub133 - sub1_sub13_sub133|sub1.sub14 - sub1_sub14)[\s\S]){3}$/);
      assert.ok(fileContents.file_sub1.match(/sub123/) && fileContents.file_sub1.match(/sub133/) && fileContents.file_sub1.match(/sub14/));
      assert.ok(!fileContents.file_sub1.match(/sub2/));

      // only catgories starting with 'sub1.sub12' and 'sub1.sub13'
      assert.match(fileContents.file_sub1_sub12$sub1_sub13, /^(\[\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{3}\] \[INFO\] (sub1.sub12.sub123 - sub1_sub12_sub123|sub1.sub13.sub133 - sub1_sub13_sub133)[\s\S]){2}$/);
      assert.ok(fileContents.file_sub1_sub12$sub1_sub13.match(/sub123/) && fileContents.file_sub1_sub12$sub1_sub13.match(/sub133/));
      assert.ok(!fileContents.file_sub1_sub12$sub1_sub13.match(/sub14|sub2/));

      // only catgories starting with 'sub1.sub12'
      assert.match(fileContents.file_sub1_sub12, /^(\[\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{3}\] \[INFO\] (sub1.sub12.sub123 - sub1_sub12_sub123)[\s\S]){1}$/);
      assert.ok(!fileContents.file_sub1_sub12.match(/sub14|sub2|sub13/));

    }
  },
  'with a max file size and no backups': {
    topic: function() {
      var testFile = path.join(__dirname, '/fa-maxFileSize-test.log')
      , logger = log4js.getLogger('max-file-size')
      , that = this;
      remove(testFile);
      remove(testFile + '.1');
      //log file of 100 bytes maximum, no backups
      log4js.clearAppenders();
      log4js.addAppender(
        require('../lib/appenders/file').appender(testFile, log4js.layouts.basicLayout, 100, 0), 
        'max-file-size'
      );
      logger.info("This is the first log message.");
      logger.info("This is an intermediate log message.");
      logger.info("This is the second log message.");
      //wait for the file system to catch up
      setTimeout(function() {
        fs.readFile(testFile, "utf8", that.callback);
      }, 100);
    },
    'log file should only contain the second message': function(err, fileContents) {
      assert.include(fileContents, "This is the second log message.");
      assert.equal(fileContents.indexOf("This is the first log message."), -1);
    },
    'the number of files': {
      topic: function() {
        fs.readdir(__dirname, this.callback);
      },
      'starting with the test file name should be two': function(err, files) {
        //there will always be one backup if you've specified a max log size
        var logFiles = files.filter(
          function(file) { return file.indexOf('fa-maxFileSize-test.log') > -1; }
        );
        assert.equal(logFiles.length, 2);
      }
    }
  },
  'with a max file size and 2 backups': {
    topic: function() {
      var testFile = path.join(__dirname, '/fa-maxFileSize-with-backups-test.log')
      , logger = log4js.getLogger('max-file-size-backups');
      remove(testFile);
      remove(testFile+'.1');
      remove(testFile+'.2');
      
      //log file of 50 bytes maximum, 2 backups
      log4js.clearAppenders();
      log4js.addAppender(
        require('../lib/appenders/file').appender(testFile, log4js.layouts.basicLayout, 50, 2), 
        'max-file-size-backups'
      );
      logger.info("This is the first log message.");
      logger.info("This is the second log message.");
      logger.info("This is the third log message.");
      logger.info("This is the fourth log message.");
      var that = this;
      //give the system a chance to open the stream
      setTimeout(function() {
        fs.readdir(__dirname, function(err, files) { 
          if (files) { 
            that.callback(null, files.sort()); 
          } else { 
            that.callback(err, files); 
          }
        });
      }, 200);
    },
    'the log files': {
      topic: function(files) {
        var logFiles = files.filter(
          function(file) { return file.indexOf('fa-maxFileSize-with-backups-test.log') > -1; }
        );
        return logFiles;
      },
      'should be 3': function (files) {
        assert.equal(files.length, 3);
      },
      'should be named in sequence': function (files) {
        assert.deepEqual(files, [
          'fa-maxFileSize-with-backups-test.log', 
          'fa-maxFileSize-with-backups-test.log.1', 
          'fa-maxFileSize-with-backups-test.log.2'
        ]);
      },
      'and the contents of the first file': {
        topic: function(logFiles) {
          fs.readFile(path.join(__dirname, logFiles[0]), "utf8", this.callback);
        },
        'should be the last log message': function(contents) {
          assert.include(contents, 'This is the fourth log message.');
        }
      },
      'and the contents of the second file': {
        topic: function(logFiles) {
          fs.readFile(path.join(__dirname, logFiles[1]), "utf8", this.callback);
        },
        'should be the third log message': function(contents) {
          assert.include(contents, 'This is the third log message.');
        }
      },
      'and the contents of the third file': {
        topic: function(logFiles) {
          fs.readFile(path.join(__dirname, logFiles[2]), "utf8", this.callback);
        },
        'should be the second log message': function(contents) {
          assert.include(contents, 'This is the second log message.');
        }
      }
    }
  },
  'with a max file size and 2 compressed backups': {
    topic: function() {
      var testFile = path.join(__dirname, '/fa-maxFileSize-with-backups-compressed-test.log')
      , logger = log4js.getLogger('max-file-size-backups');
      remove(testFile);
      remove(testFile+'.1.gz');
      remove(testFile+'.2.gz');
      
      //log file of 50 bytes maximum, 2 backups
      log4js.clearAppenders();
      log4js.addAppender(
        require('../lib/appenders/file').appender(testFile, log4js.layouts.basicLayout, 50, 2, true), 
        'max-file-size-backups'
      );
      logger.info("This is the first log message.");
      logger.info("This is the second log message.");
      logger.info("This is the third log message.");
      logger.info("This is the fourth log message.");
      var that = this;
      //give the system a chance to open the stream
      setTimeout(function() {
        fs.readdir(__dirname, function(err, files) { 
          if (files) { 
            that.callback(null, files.sort()); 
          } else { 
            that.callback(err, files); 
          }
        });
      }, 1000);
    },
    'the log files': {
      topic: function(files) {
        var logFiles = files.filter(
          function(file) { return file.indexOf('fa-maxFileSize-with-backups-compressed-test.log') > -1; }
        );
        return logFiles;
      },
      'should be 3': function (files) {
        assert.equal(files.length, 3);
      },
      'should be named in sequence': function (files) {
        assert.deepEqual(files, [
          'fa-maxFileSize-with-backups-compressed-test.log', 
          'fa-maxFileSize-with-backups-compressed-test.log.1.gz', 
          'fa-maxFileSize-with-backups-compressed-test.log.2.gz'
        ]);
      },
      'and the contents of the first file': {
        topic: function(logFiles) {
          fs.readFile(path.join(__dirname, logFiles[0]), "utf8", this.callback);
        },
        'should be the last log message': function(contents) {
          assert.include(contents, 'This is the fourth log message.');
        }
      },
      'and the contents of the second file': {
        topic: function(logFiles) {
          zlib.gunzip(fs.readFileSync(path.join(__dirname, logFiles[1])), this.callback);
        },
        'should be the third log message': function(contents) {
          assert.include(contents.toString('utf8'), 'This is the third log message.');
        }
      },
      'and the contents of the third file': {
        topic: function(logFiles) {
          zlib.gunzip(fs.readFileSync(path.join(__dirname, logFiles[2])), this.callback);
        },
        'should be the second log message': function(contents) {
          assert.include(contents.toString('utf8'), 'This is the second log message.');
        }
      }
    }
  }
}).addBatch({
  'configure' : {
    'with fileAppender': {
      topic: function() {
        var log4js = require('../lib/log4js')
        , logger;
        //this config file defines one file appender (to ./tmp-tests.log)
        //and sets the log level for "tests" to WARN
        log4js.configure('./test/log4js.json');
        logger = log4js.getLogger('tests');
        logger.info('this should not be written to the file');
        logger.warn('this should be written to the file');
        
        fs.readFile('tmp-tests.log', 'utf8', this.callback);
      },
      'should load appender configuration from a json file': function (err, contents) {
        assert.include(contents, 'this should be written to the file' + EOL);
        assert.equal(contents.indexOf('this should not be written to the file'), -1);
      }
    }
  }
}).addBatch({
  'when underlying stream errors': {
    topic: function() {
      var consoleArgs
      , errorHandler
      , fileAppender = sandbox.require(
        '../lib/appenders/file',
        {
          globals: {
            console: {
              error: function() {
                consoleArgs = Array.prototype.slice.call(arguments);
              }
            }
          },
          requires: {
            '../streams': {
              RollingFileStream: function(filename) {
                
                this.end = function() {};
                this.on = function(evt, cb) {
                  if (evt === 'error') {
                    errorHandler = cb;
                  }
                };
              }
            }
          }   
        }
      );
      fileAppender.appender('test1.log', null, 100);
      errorHandler({ error: 'aargh' });
      return consoleArgs;
    },
    'should log the error to console.error': function(consoleArgs) {
      assert.isNotEmpty(consoleArgs);
      assert.equal(consoleArgs[0], 'log4js.fileAppender - Writing to file %s, error happened ');
      assert.equal(consoleArgs[1], 'test1.log');
      assert.equal(consoleArgs[2].error, 'aargh');
    }
  }

}).export(module);
