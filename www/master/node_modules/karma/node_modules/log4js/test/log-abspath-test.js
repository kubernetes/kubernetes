"use strict";
var vows = require('vows')
, assert = require('assert')
, path = require('path')
, sandbox = require('sandboxed-module');

vows.describe('log4js-abspath').addBatch({
  'options': {
    topic: function() {
      var appenderOptions,
      log4js = sandbox.require(
        '../lib/log4js',
        { requires:
          { './appenders/fake':
            { name: "fake",
              appender: function() {},
              configure: function(configuration, options) {
                appenderOptions = options;
                return function() {};
              }
            }
          }
        }
      ),
      config = {
        "appenders": [
          {
            "type" : "fake",
            "filename" : "cheesy-wotsits.log"
          }
        ]
      };
      
      log4js.configure(config, {
        cwd: '/absolute/path/to'
      });
      return appenderOptions;
    },
    'should be passed to appenders during configuration': function(options) {
      assert.equal(options.cwd, '/absolute/path/to');
    }
  },

  'file appender': {
    topic: function() {
      var fileOpened,
      fileAppender = sandbox.require(
        '../lib/appenders/file',
        { requires:
          { '../streams':
            { RollingFileStream: 
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
      fileAppender.configure(
        { 
          filename: "whatever.log", 
          maxLogSize: 10 
        }, 
        { cwd: '/absolute/path/to' }
      );
      return fileOpened;
    },
    'should prepend options.cwd to config.filename': function(fileOpened) {
      var expected = path.sep + path.join("absolute", "path", "to", "whatever.log");
      assert.equal(fileOpened, expected);
    }
  },
}).export(module);
