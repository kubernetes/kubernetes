"use strict";
var vows = require('vows')
, assert = require('assert')
, log4js = require('../lib/log4js')
, sandbox = require('sandboxed-module')
;

function setupLogging(category, options) {
  var msgs = [];
  
  var fakeLoggly = {
    createClient: function (options) {
      return {
        config: options,
        log: function (msg, tags) {
          msgs.push({
            msg: msg,
            tags: tags
          });
        }
      };
    }
  };

  var fakeLayouts = {
    layout: function(type, config) {
      this.type = type;
      this.config = config;
      return log4js.layouts.messagePassThroughLayout;
    },
    basicLayout: log4js.layouts.basicLayout,
    messagePassThroughLayout: log4js.layouts.messagePassThroughLayout
  };

  var fakeConsole = {
    errors: [],
    error: function(msg, value) {
      this.errors.push({ msg: msg, value: value });
    }
  };

  var logglyModule = sandbox.require('../lib/appenders/loggly', {
    requires: {
      'loggly': fakeLoggly,
      '../layouts': fakeLayouts
    },
    globals: {
      console: fakeConsole
    }
  });

  log4js.addAppender(logglyModule.configure(options), category);
  
  return {
    logger: log4js.getLogger(category),
    loggly: fakeLoggly,
    layouts: fakeLayouts,
    console: fakeConsole,
    results: msgs
  };
}

log4js.clearAppenders();
vows.describe('log4js logglyAppender').addBatch({
  'minimal config': {
    topic: function() {
      var setup = setupLogging('loggly', {
        token: 'your-really-long-input-token',
        subdomain: 'your-subdomain',
        tags: ['loggly-tag1', 'loggly-tag2', 'loggly-tagn'] 
      });
      
      setup.logger.log('trace', 'Log event #1');
      return setup;
    },
    'there should be one message only': function (topic) {
      //console.log('topic', topic);
      assert.equal(topic.results.length, 1);
    }
  }

}).export(module);
