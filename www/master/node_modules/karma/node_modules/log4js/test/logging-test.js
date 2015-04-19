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

vows.describe('log4js').addBatch({

    'getBufferedLogger': {
        topic: function () {
            var log4js = require('../lib/log4js');
            log4js.clearAppenders();
            var logger = log4js.getBufferedLogger('tests');
            return logger;
        },

        'should take a category and return a logger': function (logger) {
            assert.equal(logger.target.category, 'tests');
            assert.isFunction(logger.flush);
            assert.isFunction(logger.trace);
            assert.isFunction(logger.debug);
            assert.isFunction(logger.info);
            assert.isFunction(logger.warn);
            assert.isFunction(logger.error);
            assert.isFunction(logger.fatal);
        },

        'cache events': {
            topic: function () {
                var log4js = require('../lib/log4js');
                log4js.clearAppenders();
                var logger = log4js.getBufferedLogger('tests1');
                var events = [];
                logger.target.addListener("log", function (logEvent) { events.push(logEvent); });
                logger.debug("Debug event");
                logger.trace("Trace event 1");
                logger.trace("Trace event 2");
                logger.warn("Warning event");
                logger.error("Aargh!", new Error("Pants are on fire!"));
                logger.error("Simulated CouchDB problem", { err: 127, cause: "incendiary underwear" });
                return events;
            },

            'should not emit log events if .flush() is not called.': function (events) {
                assert.equal(events.length, 0);
            }
        },

        'log events after flush() is called': {
            topic: function () {
                var log4js = require('../lib/log4js');
                log4js.clearAppenders();
                var logger = log4js.getBufferedLogger('tests2');
                logger.target.setLevel("TRACE");
                var events = [];
                logger.target.addListener("log", function (logEvent) { events.push(logEvent); });
                logger.debug("Debug event");
                logger.trace("Trace event 1");
                logger.trace("Trace event 2");
                logger.warn("Warning event");
                logger.error("Aargh!", new Error("Pants are on fire!"));
                logger.error("Simulated CouchDB problem", { err: 127, cause: "incendiary underwear" });
                logger.flush();
                return events;
            },

            'should emit log events when .flush() is called.': function (events) {
                assert.equal(events.length, 6);
            }
        }
    }, 
    
   
  'getLogger': {
    topic: function() {
      var log4js = require('../lib/log4js');
      log4js.clearAppenders();
      var logger = log4js.getLogger('tests');
      logger.setLevel("DEBUG");
      return logger;
    },
    
    'should take a category and return a logger': function(logger) {
      assert.equal(logger.category, 'tests');
      assert.equal(logger.level.toString(), "DEBUG");
      assert.isFunction(logger.debug);
      assert.isFunction(logger.info);
      assert.isFunction(logger.warn);
      assert.isFunction(logger.error);
      assert.isFunction(logger.fatal);
    },

    'log events' : {
      topic: function(logger) {
        var events = [];
        logger.addListener("log", function (logEvent) { events.push(logEvent); });
        logger.debug("Debug event");
        logger.trace("Trace event 1");
        logger.trace("Trace event 2");
        logger.warn("Warning event");
        logger.error("Aargh!", new Error("Pants are on fire!"));
        logger.error("Simulated CouchDB problem", { err: 127, cause: "incendiary underwear" });
        return events;
      },
      
      'should emit log events': function(events) {
        assert.equal(events[0].level.toString(), 'DEBUG');
        assert.equal(events[0].data[0], 'Debug event');
        assert.instanceOf(events[0].startTime, Date);
      },
      
      'should not emit events of a lower level': function(events) {
        assert.equal(events.length, 4);
        assert.equal(events[1].level.toString(), 'WARN');
      },
      
      'should include the error if passed in': function(events) {
        assert.instanceOf(events[2].data[1], Error);
        assert.equal(events[2].data[1].message, 'Pants are on fire!');
      }
    }
  },

  'when shutdown is called': {
    topic: function() {
      var events = {
       appenderShutdownCalled: false,
       shutdownCallbackCalled: false
      },
      log4js = sandbox.require(
        '../lib/log4js',
        {
          requires: {
            './appenders/file':
            {
              name: "file",
              appender: function() {},
              configure: function(configuration) {
                return function() {};
              },
              shutdown: function(cb) {
                events.appenderShutdownCalled = true;
                cb();
              }
            }
          }
        }
      ),
      shutdownCallback = function() {
        events.shutdownCallbackCalled = true;
      },
      config = { appenders:
                 [ { "type" : "file",
                     "filename" : "cheesy-wotsits.log",
                     "maxLogSize" : 1024,
                     "backups" : 3
                   }
                 ]
               };

      log4js.configure(config);
      log4js.shutdown(shutdownCallback);
      // Re-enable log writing so other tests that use logger are not
      // affected.
      require('../lib/logger').enableAllLogWrites();
      return events;
    },

    'should invoke appender shutdowns': function(events) {
      assert.ok(events.appenderShutdownCalled);
    },

    'should call callback': function(events) {
      assert.ok(events.shutdownCallbackCalled);
    }
  },
  
  'invalid configuration': {
    'should throw an exception': function() {
      assert.throws(function() {
        require('log4js').configure({ "type": "invalid" });
      });
    }
  },
  
  'configuration when passed as object': {
    topic: function() {
      var appenderConfig, 
      log4js = sandbox.require(
        '../lib/log4js', 
        { 
          requires: { 
            './appenders/file': 
            {
              name: "file",
              appender: function() {},
              configure: function(configuration) {
                appenderConfig = configuration;
                return function() {};
              }
            }
          }
        }
      ), 
      config = { appenders: 
                 [ { "type" : "file",
                     "filename" : "cheesy-wotsits.log",
                     "maxLogSize" : 1024,
                     "backups" : 3
                   }
                 ]
               };
      log4js.configure(config);
      return appenderConfig;
    },
    'should be passed to appender config': function(configuration) {
      assert.equal(configuration.filename, 'cheesy-wotsits.log');
    }
  },

  'configuration that causes an error': {
    topic: function() {
      var log4js = sandbox.require(
        '../lib/log4js', 
        { 
          requires: { 
            './appenders/file': 
            {
              name: "file",
              appender: function() {},
              configure: function(configuration) {
                throw new Error("oh noes");
              }
            }
          }
        }
      ), 
      config = { appenders: 
                 [ { "type" : "file",
                     "filename" : "cheesy-wotsits.log",
                     "maxLogSize" : 1024,
                     "backups" : 3
                   }
                 ]
               };
      try {
        log4js.configure(config);
      } catch (e) {
        return e;
      }    
    },
    'should wrap error in a meaningful message': function(e) {
      assert.ok(e.message.indexOf('log4js configuration problem for') > -1);
    }
  },

  'configuration when passed as filename': {
    topic: function() {
      var appenderConfig, 
      configFilename, 
      log4js = sandbox.require(
        '../lib/log4js', 
        { requires:
          { 'fs':
            { statSync: 
              function() {
                return { mtime: Date.now() };
              },
              readFileSync: 
              function(filename) {
                configFilename = filename;
                return JSON.stringify({
                  appenders: [
                    { type: "file"
                      , filename: "whatever.log"
                    }
                  ]
                });
              },
              readdirSync: 
              function() {
                return ['file'];
              }
            }, 
            './appenders/file': 
            { name: "file", 
              appender: function() {}, 
              configure: function(configuration) {
                appenderConfig = configuration;
                return function() {};
              }
            }
          }
        }
      );
      log4js.configure("/path/to/cheese.json");
      return [ configFilename, appenderConfig ];
    },
    'should read the config from a file': function(args) {
      assert.equal(args[0], '/path/to/cheese.json');
    },
    'should pass config to appender': function(args) {
      assert.equal(args[1].filename, "whatever.log");
    }
  },

  'with no appenders defined' : {
    topic: function() {
      var logger, 
      that = this, 
      fakeConsoleAppender = {
        name: "console", 
        appender: function() {
          return function(evt) {
            that.callback(null, evt);
          };
        }, 
        configure: function() {
          return fakeConsoleAppender.appender();
        }
      }, 
      log4js = sandbox.require(
        '../lib/log4js', 
        {
          requires: {
            './appenders/console': fakeConsoleAppender
          }
        }
      );
      logger = log4js.getLogger("some-logger");
      logger.debug("This is a test");
    },
    'should default to the console appender': function(evt) {
      assert.equal(evt.data[0], "This is a test");
    }
  },

  'addAppender' : {
    topic: function() {
      var log4js = require('../lib/log4js');
      log4js.clearAppenders();
      return log4js;
    },
    'without a category': {
      'should register the function as a listener for all loggers': function (log4js) {
        var appenderEvent, 
        appender = function(evt) { appenderEvent = evt; }, 
        logger = log4js.getLogger("tests");

        log4js.addAppender(appender);
        logger.debug("This is a test");
        assert.equal(appenderEvent.data[0], "This is a test");
        assert.equal(appenderEvent.categoryName, "tests");
        assert.equal(appenderEvent.level.toString(), "DEBUG");
      },
      'if an appender for a category is defined': {
        'should register for that category': function (log4js) {
          var otherEvent, 
          appenderEvent, 
          cheeseLogger;
          
          log4js.addAppender(function (evt) { appenderEvent = evt; });
          log4js.addAppender(function (evt) { otherEvent = evt; }, 'cheese');
        
          cheeseLogger = log4js.getLogger('cheese');
          cheeseLogger.debug('This is a test');
          assert.deepEqual(appenderEvent, otherEvent);
          assert.equal(otherEvent.data[0], 'This is a test');
          assert.equal(otherEvent.categoryName, 'cheese');
          
          otherEvent = undefined;
          appenderEvent = undefined;
          log4js.getLogger('pants').debug("this should not be propagated to otherEvent");
          assert.isUndefined(otherEvent);
          assert.equal(appenderEvent.data[0], "this should not be propagated to otherEvent");
        }
      }
    },
    
    'with a category': {
      'should only register the function as a listener for that category': function(log4js) {
        var appenderEvent, 
        appender = function(evt) { appenderEvent = evt; }, 
        logger = log4js.getLogger("tests");

        log4js.addAppender(appender, 'tests');
        logger.debug('this is a category test');
        assert.equal(appenderEvent.data[0], 'this is a category test');
        
        appenderEvent = undefined;
        log4js.getLogger('some other category').debug('Cheese');
        assert.isUndefined(appenderEvent);
      }
    },
    
    'with multiple categories': {
      'should register the function as a listener for all the categories': function(log4js) {
        var appenderEvent, 
        appender = function(evt) { appenderEvent = evt; }, 
        logger = log4js.getLogger('tests');

        log4js.addAppender(appender, 'tests', 'biscuits');
        
        logger.debug('this is a test');
        assert.equal(appenderEvent.data[0], 'this is a test');
        appenderEvent = undefined;
        
        var otherLogger = log4js.getLogger('biscuits');
        otherLogger.debug("mmm... garibaldis");
        assert.equal(appenderEvent.data[0], "mmm... garibaldis");
        
        appenderEvent = undefined;
        
        log4js.getLogger("something else").debug("pants");
        assert.isUndefined(appenderEvent);
      },
      'should register the function when the list of categories is an array': function(log4js) {
        var appenderEvent, 
        appender = function(evt) { appenderEvent = evt; };

        log4js.addAppender(appender, ['tests', 'pants']);
        
        log4js.getLogger('tests').debug('this is a test');
        assert.equal(appenderEvent.data[0], 'this is a test');
        
        appenderEvent = undefined;
        
        log4js.getLogger('pants').debug("big pants");
        assert.equal(appenderEvent.data[0], "big pants");
        
        appenderEvent = undefined;

        log4js.getLogger("something else").debug("pants");
        assert.isUndefined(appenderEvent);
      }
    }
  },
  
  'default setup': {
    topic: function() {
      var appenderEvents = [],
      fakeConsole = {
        'name': 'console', 
        'appender': function () {
          return function(evt) {
            appenderEvents.push(evt);
          };
        }, 
        'configure': function (config) {
          return fakeConsole.appender();
        }
      },
      globalConsole = {
        log: function() { }
      },
      log4js = sandbox.require(
        '../lib/log4js',
        {
          requires: {
            './appenders/console': fakeConsole
          },
          globals: {
            console: globalConsole
          }
        }
      ),
      logger = log4js.getLogger('a-test');
      
      logger.debug("this is a test");
      globalConsole.log("this should not be logged");
      
      return appenderEvents;
    },
    
    'should configure a console appender': function(appenderEvents) {
      assert.equal(appenderEvents[0].data[0], 'this is a test');
    },
    
    'should not replace console.log with log4js version': function(appenderEvents) {
      assert.equal(appenderEvents.length, 1);
    }
  },
  
  'console' : {
    topic: setupConsoleTest,
    
    'when replaceConsole called': {
      topic: function(test) {
        test.log4js.replaceConsole();
        
        test.fakeConsole.log("Some debug message someone put in a module");
        test.fakeConsole.debug("Some debug");
        test.fakeConsole.error("An error");
        test.fakeConsole.info("some info");
        test.fakeConsole.warn("a warning");
        
        test.fakeConsole.log("cheese (%s) and biscuits (%s)", "gouda", "garibaldis");
        test.fakeConsole.log({ lumpy: "tapioca" });
        test.fakeConsole.log("count %d", 123);
        test.fakeConsole.log("stringify %j", { lumpy: "tapioca" });
        
        return test.logEvents;
      },
      
      'should replace console.log methods with log4js ones': function(logEvents) {
        assert.equal(logEvents.length, 9);
        assert.equal(logEvents[0].data[0], "Some debug message someone put in a module");
        assert.equal(logEvents[0].level.toString(), "INFO");
        assert.equal(logEvents[1].data[0], "Some debug");
        assert.equal(logEvents[1].level.toString(), "DEBUG");
        assert.equal(logEvents[2].data[0], "An error");
        assert.equal(logEvents[2].level.toString(), "ERROR");
        assert.equal(logEvents[3].data[0], "some info");
        assert.equal(logEvents[3].level.toString(), "INFO");
        assert.equal(logEvents[4].data[0], "a warning");
        assert.equal(logEvents[4].level.toString(), "WARN");
        assert.equal(logEvents[5].data[0], "cheese (%s) and biscuits (%s)");
        assert.equal(logEvents[5].data[1], "gouda");
        assert.equal(logEvents[5].data[2], "garibaldis");
      }
    },
    'when turned off': {
      topic: function(test) {
        test.log4js.restoreConsole();
        try {
          test.fakeConsole.log("This should cause the error described in the setup");
        } catch (e) {
          return e;
        }
      },
      'should call the original console methods': function (err) {
        assert.instanceOf(err, Error);
        assert.equal(err.message, "this should not be called.");
      }
    }
  },
  'console configuration': {
    topic: setupConsoleTest,
    'when disabled': {
      topic: function(test) {
        test.log4js.replaceConsole();
        test.log4js.configure({ replaceConsole: false });
        try {
          test.fakeConsole.log("This should cause the error described in the setup");
        } catch (e) {
          return e;
        }
      },
      'should allow for turning off console replacement': function (err) {
        assert.instanceOf(err, Error);
        assert.equal(err.message, 'this should not be called.');
      }
    },
    'when enabled': {
      topic: function(test) {
        test.log4js.restoreConsole();
        test.log4js.configure({ replaceConsole: true });
        //log4js.configure clears all appenders
        test.log4js.addAppender(function(evt) {
          test.logEvents.push(evt);
        });

        test.fakeConsole.debug("Some debug");
        return test.logEvents;
      },
      
      'should allow for turning on console replacement': function (logEvents) {
        assert.equal(logEvents.length, 1);
        assert.equal(logEvents[0].level.toString(), "DEBUG");
        assert.equal(logEvents[0].data[0], "Some debug");
      }
    }
  },
  'configuration persistence' : {
    topic: function() {
      var logEvent,
      firstLog4js = require('../lib/log4js'),
      secondLog4js;
      
      firstLog4js.clearAppenders();
      firstLog4js.addAppender(function(evt) { logEvent = evt; });
      
      secondLog4js = require('../lib/log4js');
      secondLog4js.getLogger().info("This should go to the appender defined in firstLog4js");
      
      return logEvent;
    },
    'should maintain appenders between requires': function (logEvent) {
      assert.equal(logEvent.data[0], "This should go to the appender defined in firstLog4js");
    }
  },

  'getDefaultLogger': {
    topic: function() {
      return require('../lib/log4js').getDefaultLogger();
    },
    'should return a logger': function(logger) {
      assert.ok(logger.info);
      assert.ok(logger.debug);
      assert.ok(logger.error);
    }
  }
}).export(module);
