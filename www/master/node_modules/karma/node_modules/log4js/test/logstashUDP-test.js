"use strict";
var sys = require("sys");
var vows = require('vows')
, assert = require('assert')
, log4js = require('../lib/log4js')
, sandbox = require('sandboxed-module')
;

function setupLogging(category, options) {
  var udpSent = {};
  
  var fakeDgram = {
    createSocket: function (type) {
      return {
        send: function(buffer, offset, length, port, host, callback) {
          udpSent.date = new Date();
          udpSent.host = host;
          udpSent.port = port;
          udpSent.length = length;
          udpSent.offset = 0;
          udpSent.buffer = buffer;
          callback(undefined, length);
        }
      };
    }
  };

  var logstashModule = sandbox.require('../lib/appenders/logstashUDP', {
    requires: {
      'dgram': fakeDgram
    }
  });
  log4js.clearAppenders();
  log4js.addAppender(logstashModule.configure(options), category);
  
  return {
    logger: log4js.getLogger(category),
    results: udpSent
  };
}

vows.describe('logstashUDP appender').addBatch({
  'when logging with logstash via UDP': {
    topic: function() {
      var setup = setupLogging('logstashUDP', {
        "host": "127.0.0.1",
        "port": 10001,
        "type": "logstashUDP",
        "logType": "myAppType",
        "category": "myLogger",
        "fields": {
          "field1": "value1",
          "field2": "value2"
        },
        "layout": {
          "type": "pattern",
          "pattern": "%m"
        }
      });
      setup.logger.log('trace', 'Log event #1');
      return setup;
    },
    'an UDP packet should be sent': function (topic) {
      assert.equal(topic.results.host, "127.0.0.1");
      assert.equal(topic.results.port, 10001);
      assert.equal(topic.results.offset, 0);
      var json = JSON.parse(topic.results.buffer.toString());
      assert.equal(json.type, 'myAppType');
      var fields = {
        field1: 'value1',
        field2: 'value2',
        level: 'TRACE'
      };
      assert.equal(JSON.stringify(json.fields), JSON.stringify(fields));
      assert.equal(json.message, 'Log event #1');
      // Assert timestamp, up to hours resolution.
      var date = new Date(json['@timestamp']);
      assert.equal(
        date.toISOString().substring(0, 14),
        topic.results.date.toISOString().substring(0, 14)
      );
    }
  },

  'when missing some options': {
    topic: function() {
      var setup = setupLogging('myLogger', {
        "host": "127.0.0.1",
        "port": 10001,
        "type": "logstashUDP",
        "category": "myLogger",
        "layout": {
          "type": "pattern",
          "pattern": "%m"
        }
      });
      setup.logger.log('trace', 'Log event #1');
      return setup;
    },
    'it sets some defaults': function (topic) {
      var json = JSON.parse(topic.results.buffer.toString());
      assert.equal(json.type, 'myLogger');
      assert.equal(JSON.stringify(json.fields), JSON.stringify({'level': 'TRACE'}));
    }
  }
}).export(module);
