"use strict";
var vows = require('vows')
, assert = require('assert')
, sandbox = require('sandboxed-module');

function setupConsoleTest() {
  var fakeConsole = {}
  , logEvents = []
  , log4js;
  
  ['trace','debug','log','info','warn','error'].forEach(function(fn) {
    fakeConsole[fn] = function() {
      throw new Error("this should not be called.");
    };
  });

  log4js = sandbox.require(
    '../lib/log4js', 
    {
      globals: {
        console: fakeConsole
      }
    }
  );

  log4js.clearAppenders();
  log4js.addAppender(function(evt) {
    logEvents.push(evt);
  });

  return { log4js: log4js, logEvents: logEvents, fakeConsole: fakeConsole };
}

vows.describe('reload configuration').addBatch({
  'with config file changing' : {
    topic: function() {
      var pathsChecked = [],
      logEvents = [],
      logger,
      modulePath = 'path/to/log4js.json',
      fakeFS = {
        lastMtime: Date.now(),
        config: { 
          appenders: [ 
            { type: 'console', layout: { type: 'messagePassThrough' } } 
          ],
          levels: { 'a-test' : 'INFO' } 
        },
        readFileSync: function (file, encoding) {
          assert.equal(file, modulePath);
          assert.equal(encoding, 'utf8');
          return JSON.stringify(fakeFS.config);
        },
        statSync: function (path) {
          pathsChecked.push(path);
          if (path === modulePath) {
            fakeFS.lastMtime += 1;
            return { mtime: new Date(fakeFS.lastMtime) };
          } else {
            throw new Error("no such file");
          }
        }
      },
      fakeConsole = {
        'name': 'console',
        'appender': function () {
          return function(evt) { logEvents.push(evt); };
        },
        'configure': function (config) {
          return fakeConsole.appender();
        }
      },
      setIntervalCallback,
      fakeSetInterval = function(cb, timeout) {
        setIntervalCallback = cb;
      },
      log4js = sandbox.require(
        '../lib/log4js',
        {
          requires: {
            'fs': fakeFS,
            './appenders/console': fakeConsole
          },
          globals: {
            'console': fakeConsole,
            'setInterval' : fakeSetInterval,
          }
        }
      );
      
      log4js.configure('path/to/log4js.json', { reloadSecs: 30 });
      logger = log4js.getLogger('a-test');
      logger.info("info1");
      logger.debug("debug2 - should be ignored");
      fakeFS.config.levels['a-test'] = "DEBUG";
      setIntervalCallback();
      logger.info("info3");
      logger.debug("debug4");
      
      return logEvents;
    },
    'should configure log4js from first log4js.json found': function(logEvents) {
      assert.equal(logEvents[0].data[0], 'info1');
      assert.equal(logEvents[1].data[0], 'info3');
      assert.equal(logEvents[2].data[0], 'debug4');
      assert.equal(logEvents.length, 3);
    }
  },
  
  'with config file staying the same' : {
    topic: function() {
      var pathsChecked = [],
      fileRead = 0,
      logEvents = [],
      logger,
      modulePath = require('path').normalize(__dirname + '/../lib/log4js.json'),
      mtime = new Date(),
      fakeFS = {
        config: { 
          appenders: [ 
            { type: 'console', layout: { type: 'messagePassThrough' } } 
          ],
          levels: { 'a-test' : 'INFO' } 
        },
        readFileSync: function (file, encoding) {
          fileRead += 1;
          assert.isString(file);
          assert.equal(file, modulePath);
          assert.equal(encoding, 'utf8');
          return JSON.stringify(fakeFS.config);
        },
        statSync: function (path) {
          pathsChecked.push(path);
          if (path === modulePath) {
            return { mtime: mtime };
          } else {
            throw new Error("no such file");
          }
        }
      },
      fakeConsole = {
        'name': 'console',
        'appender': function () {
          return function(evt) { logEvents.push(evt); };
        },
        'configure': function (config) {
          return fakeConsole.appender();
        }
      },
      setIntervalCallback,
      fakeSetInterval = function(cb, timeout) {
        setIntervalCallback = cb;
      },
      log4js = sandbox.require(
        '../lib/log4js',
        {
          requires: {
            'fs': fakeFS,
            './appenders/console': fakeConsole
          },
          globals: {
            'console': fakeConsole,
            'setInterval' : fakeSetInterval,
          }
        }
      );
      
      log4js.configure(modulePath, { reloadSecs: 3 });
      logger = log4js.getLogger('a-test');
      logger.info("info1");
      logger.debug("debug2 - should be ignored");
      setIntervalCallback();
      logger.info("info3");
      logger.debug("debug4");
      
      return [ pathsChecked, logEvents, modulePath, fileRead ];
    },
    'should only read the configuration file once': function(args) {
      var fileRead = args[3];
      assert.equal(fileRead, 1);
    },
    'should configure log4js from first log4js.json found': function(args) {
      var logEvents = args[1];
      assert.equal(logEvents.length, 2);
      assert.equal(logEvents[0].data[0], 'info1');
      assert.equal(logEvents[1].data[0], 'info3');
    }
  },

  'when config file is removed': {
    topic: function() {
      var pathsChecked = [],
      fileRead = 0,
      logEvents = [],
      logger,
      modulePath = require('path').normalize(__dirname + '/../lib/log4js.json'),
      mtime = new Date(),
      fakeFS = {
        config: { 
          appenders: [ 
            { type: 'console', layout: { type: 'messagePassThrough' } } 
          ],
          levels: { 'a-test' : 'INFO' } 
        },
        readFileSync: function (file, encoding) {
          fileRead += 1;
          assert.isString(file);
          assert.equal(file, modulePath);
          assert.equal(encoding, 'utf8');
          return JSON.stringify(fakeFS.config);
        },
        statSync: function (path) {
          this.statSync = function() {
            throw new Error("no such file");
          };
          return { mtime: new Date() };
        }
      },
      fakeConsole = {
        'name': 'console',
        'appender': function () {
          return function(evt) { logEvents.push(evt); };
        },
        'configure': function (config) {
          return fakeConsole.appender();
        }
      },
      setIntervalCallback,
      fakeSetInterval = function(cb, timeout) {
        setIntervalCallback = cb;
      },
      log4js = sandbox.require(
        '../lib/log4js',
        {
          requires: {
            'fs': fakeFS,
            './appenders/console': fakeConsole
          },
          globals: {
            'console': fakeConsole,
            'setInterval' : fakeSetInterval,
          }
        }
      );
      
      log4js.configure(modulePath, { reloadSecs: 3 });
      logger = log4js.getLogger('a-test');
      logger.info("info1");
      logger.debug("debug2 - should be ignored");
      setIntervalCallback();
      logger.info("info3");
      logger.debug("debug4");
      
      return [ pathsChecked, logEvents, modulePath, fileRead ];
    },
    'should only read the configuration file once': function(args) {
      var fileRead = args[3];
      assert.equal(fileRead, 1);
    },
    'should not clear configuration when config file not found': function(args) {
      var logEvents = args[1];
      assert.equal(logEvents.length, 3);
      assert.equal(logEvents[0].data[0], 'info1');
      assert.equal(logEvents[1].level.toString(), 'WARN');
      assert.include(logEvents[1].data[0], 'Failed to load configuration file');
      assert.equal(logEvents[2].data[0], 'info3');
    }
  },

  'when passed an object': {
    topic: function() {
      var test = setupConsoleTest();
      test.log4js.configure({}, { reloadSecs: 30 });
      return test.logEvents;
    },
    'should log a warning': function(events) {
      assert.equal(events[0].level.toString(), 'WARN');
      assert.equal(
        events[0].data[0], 
        'Ignoring configuration reload parameter for "object" configuration.'
      );
    }
  },

  'when called twice with reload options': {
    topic: function() {
      var modulePath = require('path').normalize(__dirname + '/../lib/log4js.json'),
      fakeFS = {
        readFileSync: function (file, encoding) {
          return JSON.stringify({});
        },
        statSync: function (path) {
          return { mtime: new Date() };
        }
      },
      fakeConsole = {
        'name': 'console',
        'appender': function () {
          return function(evt) { };
        },
        'configure': function (config) {
          return fakeConsole.appender();
        }
      },
      setIntervalCallback,
      intervalCleared = false,
      clearedId,
      fakeSetInterval = function(cb, timeout) {
        setIntervalCallback = cb;
        return 1234;
      },
      log4js = sandbox.require(
        '../lib/log4js',
        {
          requires: {
            'fs': fakeFS,
            './appenders/console': fakeConsole
          },
          globals: {
            'console': fakeConsole,
            'setInterval' : fakeSetInterval,
            'clearInterval': function(interval) {
              intervalCleared = true;
              clearedId = interval;
            }
          }
        }
      );
      
      log4js.configure(modulePath, { reloadSecs: 3 });
      log4js.configure(modulePath, { reloadSecs: 15 });
      
      return { cleared: intervalCleared, id: clearedId };
    },
    'should clear the previous interval': function(result) {
      assert.isTrue(result.cleared);
      assert.equal(result.id, 1234);
    }
  }
}).exportTo(module);
