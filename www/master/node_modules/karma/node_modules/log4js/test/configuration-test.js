"use strict";
var assert = require('assert')
, vows = require('vows')
, sandbox = require('sandboxed-module');

function makeTestAppender() {
  return {
    configure: function(config, options) {
      this.configureCalled = true;
      this.config = config;
      this.options = options;
      return this.appender();
    },
    appender: function() {
      var self = this;
      return function(logEvt) { self.logEvt = logEvt; };
    }
  };
}

vows.describe('log4js configure').addBatch({
  'appenders': {
    'when specified by type': {
      topic: function() {
        var testAppender = makeTestAppender(),
        log4js = sandbox.require(
          '../lib/log4js',
          {
            requires: {
              './appenders/cheese': testAppender
            }
          }
        );
        log4js.configure(
          {
            appenders: [
              { type: "cheese", flavour: "gouda" }
            ]
          },
          { pants: "yes" }
        );
        return testAppender;
      },
      'should load appender': function(testAppender) {
        assert.ok(testAppender.configureCalled);
      },
      'should pass config to appender': function(testAppender) {
        assert.equal(testAppender.config.flavour, 'gouda');
      },
      'should pass log4js options to appender': function(testAppender) {
        assert.equal(testAppender.options.pants, 'yes');
      }
    },
    'when core appender loaded via loadAppender': {
      topic: function() {
        var testAppender = makeTestAppender(),
        log4js = sandbox.require(
          '../lib/log4js',
          { requires: { './appenders/cheese': testAppender } }
        );
        
        log4js.loadAppender('cheese');
        return log4js;
      },
      'should load appender from ../lib/appenders': function(log4js) {
        assert.ok(log4js.appenders.cheese);
      },
      'should add appender configure function to appenderMakers' : function(log4js) {
        assert.isFunction(log4js.appenderMakers.cheese);
      }
    },
    'when appender in node_modules loaded via loadAppender': {
      topic: function() {
        var testAppender = makeTestAppender(),
        log4js = sandbox.require(
          '../lib/log4js',
          { requires: { 'some/other/external': testAppender } }
        );
        log4js.loadAppender('some/other/external');
        return log4js;
      },
      'should load appender via require': function(log4js) {
        assert.ok(log4js.appenders['some/other/external']);
      },
      'should add appender configure function to appenderMakers': function(log4js) {
        assert.isFunction(log4js.appenderMakers['some/other/external']);
      }
    },
    'when appender object loaded via loadAppender': {
      topic: function() {
        var testAppender = makeTestAppender(),
            log4js = sandbox.require('../lib/log4js');

        log4js.loadAppender('some/other/external', testAppender);
        return log4js;
      },
      'should load appender with provided object': function(log4js) {
        assert.ok(log4js.appenders['some/other/external']);
      },
      'should add appender configure function to appenderMakers': function(log4js) {
        assert.isFunction(log4js.appenderMakers['some/other/external']);
      }
    },
    'when configuration file loaded via LOG4JS_CONFIG environment variable': {
      topic: function() {
        process.env.LOG4JS_CONFIG = 'some/path/to/mylog4js.json';
        var fileRead = 0,
        modulePath = 'some/path/to/mylog4js.json',
        pathsChecked = [],
        mtime = new Date(),
        fakeFS = {
          config: { appenders: [ { type: 'console', layout: { type: 'messagePassThrough' } } ],
                    levels: { 'a-test' : 'INFO' } },
          readdirSync: function(dir) {
            return require('fs').readdirSync(dir);
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
        log4js = sandbox.require(
          '../lib/log4js',
          {
            requires: {
              'fs': fakeFS,
            }
          }
        );
        delete process.env.LOG4JS_CONFIG;
        return fileRead;
      },
      'should load the specified local configuration file' : function(fileRead) {
        assert.equal(fileRead, 1);
      }
    }
  }
}).exportTo(module);
