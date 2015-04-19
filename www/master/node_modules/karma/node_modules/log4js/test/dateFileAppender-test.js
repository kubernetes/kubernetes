"use strict";
var vows = require('vows')
, assert = require('assert')
, path = require('path')
, fs = require('fs')
, sandbox = require('sandboxed-module')
, log4js = require('../lib/log4js')
, EOL = require('os').EOL || '\n';

function removeFile(filename) {
  return function() {
    fs.unlink(path.join(__dirname, filename), function(err) {
      if (err) {
        console.log("Could not delete ", filename, err);
      }
    });
  };
}

vows.describe('../lib/appenders/dateFile').addBatch({
  'appender': {
    'adding multiple dateFileAppenders': {
      topic: function () {
        var listenersCount = process.listeners('exit').length,
        dateFileAppender = require('../lib/appenders/dateFile'),
        count = 5,
        logfile;
        
        while (count--) {
          logfile = path.join(__dirname, 'datefa-default-test' + count + '.log');
          log4js.addAppender(dateFileAppender.appender(logfile));
        }
        
        return listenersCount;
      },
      teardown: function() {
        removeFile('datefa-default-test0.log')();
        removeFile('datefa-default-test1.log')();
        removeFile('datefa-default-test2.log')();
        removeFile('datefa-default-test3.log')();
        removeFile('datefa-default-test4.log')();
      },
      
      'should only add one `exit` listener': function (initialCount) {
        assert.equal(process.listeners('exit').length, initialCount + 1);
      },

    },

    'exit listener': {
      topic: function() {
        var exitListener
        , openedFiles = []
        , dateFileAppender = sandbox.require(
          '../lib/appenders/dateFile',
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
                DateRollingFileStream: function(filename) {
                  openedFiles.push(filename);

                  this.end = function() {
                    openedFiles.shift();
                  };
                }
              }
            }   
          }
        );
        for (var i=0; i < 5; i += 1) {
          dateFileAppender.appender('test' + i);
        }
        assert.isNotEmpty(openedFiles);
        exitListener();
        return openedFiles;
      },
      'should close all open files': function(openedFiles) {
        assert.isEmpty(openedFiles);
      }
    },
    
    'with default settings': {
      topic: function() {
        var that = this,
        testFile = path.join(__dirname, 'date-appender-default.log'),
        appender = require('../lib/appenders/dateFile').appender(testFile),
        logger = log4js.getLogger('default-settings');
        log4js.clearAppenders();
        log4js.addAppender(appender, 'default-settings');
        
        logger.info("This should be in the file.");
        
        setTimeout(function() {
          fs.readFile(testFile, "utf8", that.callback);
        }, 100);
        
      },
      teardown: removeFile('date-appender-default.log'),
      
      'should write to the file': function(contents) {
        assert.include(contents, 'This should be in the file');
      },
      
      'should use the basic layout': function(contents) {
        assert.match(
          contents, 
            /\[\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{3}\] \[INFO\] default-settings - /
        );
      }
    }
    
  }
}).addBatch({
  'configure': {
    'with dateFileAppender': {
      topic: function() {
        var log4js = require('../lib/log4js')
        , logger;
        //this config file defines one file appender (to ./date-file-test.log)
        //and sets the log level for "tests" to WARN
        log4js.configure('test/with-dateFile.json');
        logger = log4js.getLogger('tests');
        logger.info('this should not be written to the file');
        logger.warn('this should be written to the file');
        
        fs.readFile(path.join(__dirname, 'date-file-test.log'), 'utf8', this.callback);
      },
      teardown: removeFile('date-file-test.log'),
      
      'should load appender configuration from a json file': function(err, contents) {
        if (err) {
          throw err;
        }
        assert.include(contents, 'this should be written to the file' + EOL);
        assert.equal(contents.indexOf('this should not be written to the file'), -1);
      }
    },
    'with options.alwaysIncludePattern': {
      topic: function() {
        var self = this
        , log4js = require('../lib/log4js')
        , format = require('../lib/date_format')
        , logger
        , options = {
          "appenders": [
            {
              "category": "tests", 
              "type": "dateFile", 
              "filename": "test/date-file-test",
              "pattern": "-from-MM-dd.log",
              "alwaysIncludePattern": true,
              "layout": { 
                "type": "messagePassThrough" 
              }
            }
          ]
        }
        , thisTime = format.asString(options.appenders[0].pattern, new Date());
        fs.writeFileSync(
          path.join(__dirname, 'date-file-test' + thisTime), 
          "this is existing data" + EOL,
          'utf8'
        );
        log4js.clearAppenders();
        log4js.configure(options);
        logger = log4js.getLogger('tests');
        logger.warn('this should be written to the file with the appended date');
        this.teardown = removeFile('date-file-test' + thisTime);
        //wait for filesystem to catch up
        setTimeout(function() {
          fs.readFile(path.join(__dirname, 'date-file-test' + thisTime), 'utf8', self.callback);
        }, 100);
      },
      'should create file with the correct pattern': function(contents) {
        assert.include(contents, 'this should be written to the file with the appended date');
      },
      'should not overwrite the file on open (bug found in issue #132)': function(contents) {
        assert.include(contents, 'this is existing data');
      }
    },
    'with cwd option': {
      topic: function () {
        var fileOpened,
        appender = sandbox.require(
          '../lib/appenders/dateFile',
          { requires:
            { '../streams':
              { DateRollingFileStream: 
                function(file) {
                  fileOpened = file;
                  return {
                    on: function() {},
                    end: function() {}
                  };
                }
              }
            }
          }
        );
        appender.configure(
          { 
            filename: "whatever.log", 
            maxLogSize: 10 
          }, 
          { cwd: '/absolute/path/to' }
        );
        return fileOpened;
      },
      'should prepend options.cwd to config.filename': function (fileOpened) {
        var expected = path.sep + path.join("absolute", "path", "to", "whatever.log");
        assert.equal(fileOpened, expected);
      }
    }
 
  }
}).exportTo(module);
