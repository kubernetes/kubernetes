"use strict";
var vows = require('vows')
, fs = require('fs')
, assert = require('assert')
, os = require('os')
, EOL = require('os').EOL || '\n';

function remove(filename) {
  try {
    fs.unlinkSync(filename);
  } catch (e) {
    //doesn't really matter if it failed
  }
}

vows.describe('log4js logLevelFilter').addBatch({
  'appender': {
    topic: function() {
      var log4js = require('../lib/log4js'), logEvents = [], logger;
      log4js.clearAppenders();
      log4js.addAppender(
        require('../lib/appenders/logLevelFilter')
          .appender(
            'ERROR',
            undefined,
            function(evt) { logEvents.push(evt); }
          ),
        "logLevelTest"
      );

      logger = log4js.getLogger("logLevelTest");
      logger.debug('this should not trigger an event');
      logger.warn('neither should this');
      logger.error('this should, though');
      logger.fatal('so should this');
      return logEvents;
    },
    'should only pass log events greater than or equal to its own level' : function(logEvents) {
      assert.equal(logEvents.length, 2);
      assert.equal(logEvents[0].data[0], 'this should, though');
      assert.equal(logEvents[1].data[0], 'so should this');
    }
  },

  'configure': {
    topic: function() {
      var log4js = require('../lib/log4js')
      , logger;

      remove(__dirname + '/logLevelFilter.log');
      remove(__dirname + '/logLevelFilter-warnings.log');
      remove(__dirname + '/logLevelFilter-debugs.log');

      log4js.configure('test/with-logLevelFilter.json');
      logger = log4js.getLogger("tests");
      logger.debug('debug');
      logger.info('info');
      logger.error('error');
      logger.warn('warn');
      logger.debug('debug');
      logger.trace('trace');
      //wait for the file system to catch up
      setTimeout(this.callback, 500);
    },
    'tmp-tests.log': {
      topic: function() {
        fs.readFile(__dirname + '/logLevelFilter.log', 'utf8', this.callback);
      },
      'should contain all log messages': function (contents) {
        var messages = contents.trim().split(EOL);
        assert.deepEqual(messages, ['debug','info','error','warn','debug','trace']);
      }
    },
    'tmp-tests-warnings.log': {
      topic: function() {
        fs.readFile(__dirname + '/logLevelFilter-warnings.log','utf8',this.callback);
      },
      'should contain only error and warning log messages': function(contents) {
        var messages = contents.trim().split(EOL);
        assert.deepEqual(messages, ['error','warn']);
      }
    },
    'tmp-tests-debugs.log': {
      topic: function() {
        fs.readFile(__dirname + '/logLevelFilter-debugs.log','utf8',this.callback);
      },
      'should contain only trace and debug log messages': function(contents) {
        var messages = contents.trim().split(EOL);
        assert.deepEqual(messages, ['debug','debug','trace']);
      }
    }
  }
}).export(module);
