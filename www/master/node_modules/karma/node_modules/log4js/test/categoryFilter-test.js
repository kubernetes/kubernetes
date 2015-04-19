'use strict';

var vows = require('vows')
, fs = require('fs')
, assert = require('assert')
, EOL = require('os').EOL || '\n';

function remove(filename) {
  try {
    fs.unlinkSync(filename);
  } catch (e) {
    //doesn't really matter if it failed
  }
}

vows.describe('log4js categoryFilter').addBatch({
  'appender': {
    topic: function() {

      var log4js = require('../lib/log4js'), logEvents = [], webLogger, appLogger;
      log4js.clearAppenders();
      var appender = require('../lib/appenders/categoryFilter')
        .appender(
          ['app'],
          function(evt) { logEvents.push(evt); }
        );
      log4js.addAppender(appender, ["app","web"]);

      webLogger = log4js.getLogger("web");
      appLogger = log4js.getLogger("app");

      webLogger.debug('This should get logged');
      appLogger.debug('This should not');
      webLogger.debug('Hello again');
      log4js.getLogger('db').debug('This shouldn\'t be included by the appender anyway');

      return logEvents;
    },
    'should only pass matching category' : function(logEvents) {
      assert.equal(logEvents.length, 2);
      assert.equal(logEvents[0].data[0], 'This should get logged');
      assert.equal(logEvents[1].data[0], 'Hello again');
    }
  },

  'configure': {
    topic: function() {
      var log4js = require('../lib/log4js')
      , logger, weblogger;

      remove(__dirname + '/categoryFilter-web.log');
      remove(__dirname + '/categoryFilter-noweb.log');

      log4js.configure('test/with-categoryFilter.json');
      logger = log4js.getLogger("app");
      weblogger = log4js.getLogger("web");

      logger.info('Loading app');
      logger.debug('Initialising indexes');
      weblogger.info('00:00:00 GET / 200');
      weblogger.warn('00:00:00 GET / 500');
      //wait for the file system to catch up
      setTimeout(this.callback, 500);
    },
    'tmp-tests.log': {
      topic: function() {
        fs.readFile(__dirname + '/categoryFilter-noweb.log', 'utf8', this.callback);
      },
      'should contain all log messages': function(contents) {
        var messages = contents.trim().split(EOL);
        assert.deepEqual(messages, ['Loading app','Initialising indexes']);
      }
    },
    'tmp-tests-web.log': {
      topic: function() {
        fs.readFile(__dirname + '/categoryFilter-web.log','utf8',this.callback);
      },
      'should contain only error and warning log messages': function(contents) {
        var messages = contents.trim().split(EOL);
        assert.deepEqual(messages, ['00:00:00 GET / 200','00:00:00 GET / 500']);
      }
    }
  }
}).export(module);
