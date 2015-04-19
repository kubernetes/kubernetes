"use strict";
var vows = require('vows')
, assert = require('assert')
, levels = require('../lib/levels')
, loggerModule = require('../lib/logger')
, Logger = loggerModule.Logger;

vows.describe('../lib/logger').addBatch({
  'constructor with no parameters': {
    topic: new Logger(),
    'should use default category': function(logger) {
      assert.equal(logger.category, Logger.DEFAULT_CATEGORY);
    },
    'should use TRACE log level': function(logger) {
      assert.equal(logger.level, levels.TRACE);
    }
  },

  'constructor with category': {
    topic: new Logger('cheese'),
    'should use category': function(logger) {
      assert.equal(logger.category, 'cheese');
    },
    'should use TRACE log level': function(logger) {
      assert.equal(logger.level, levels.TRACE);
    }
  },

  'constructor with category and level': {
    topic: new Logger('cheese', 'debug'),
    'should use category': function(logger) {
      assert.equal(logger.category, 'cheese');
    },
    'should use level': function(logger) {
      assert.equal(logger.level, levels.DEBUG);
    }
  },

  'isLevelEnabled': {
    topic: new Logger('cheese', 'info'),
    'should provide a level enabled function for all levels': function(logger) {
      assert.isFunction(logger.isTraceEnabled);
      assert.isFunction(logger.isDebugEnabled);
      assert.isFunction(logger.isInfoEnabled);
      assert.isFunction(logger.isWarnEnabled);
      assert.isFunction(logger.isErrorEnabled);
      assert.isFunction(logger.isFatalEnabled);
    },
    'should return the right values': function(logger) {
      assert.isFalse(logger.isTraceEnabled());
      assert.isFalse(logger.isDebugEnabled());
      assert.isTrue(logger.isInfoEnabled());
      assert.isTrue(logger.isWarnEnabled());
      assert.isTrue(logger.isErrorEnabled());
      assert.isTrue(logger.isFatalEnabled());
    }
  },

  'should emit log events': {
    topic: function() {
      var events = [],
          logger = new Logger();
      logger.addListener('log', function (logEvent) { events.push(logEvent); });
      logger.debug('Event 1');
      loggerModule.disableAllLogWrites();
      logger.debug('Event 2');
      loggerModule.enableAllLogWrites();
      logger.debug('Event 3');
      return events;
    },

    'when log writes are enabled': function(events) {
      assert.equal(events[0].data[0], 'Event 1');
    },

    'but not when log writes are disabled': function(events) {
      assert.equal(events.length, 2);
      assert.equal(events[1].data[0], 'Event 3');
    }
  }
}).exportTo(module);
