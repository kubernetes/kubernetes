"use strict";
var vows = require('vows')
, sandbox = require('sandboxed-module')
, assert = require('assert')
;

function makeFakeNet() {
  return {
    logEvents: [],
    data: [],
    cbs: {},
    createConnectionCalled: 0,
    fakeAppender: function(logEvent) {
      this.logEvents.push(logEvent);
    },
    createConnection: function(port, host) {
      var fakeNet = this;
      this.port = port;
      this.host = host;
      this.createConnectionCalled += 1;
      return {
        on: function(evt, cb) {
          fakeNet.cbs[evt] = cb;
        },
        write: function(data, encoding) {
          fakeNet.data.push(data);
          fakeNet.encoding = encoding;
        },
        end: function() {
          fakeNet.closeCalled = true;
        }
      };
    },
    createServer: function(cb) {
      var fakeNet = this;
      cb({
        remoteAddress: '1.2.3.4',
        remotePort: '1234',
        setEncoding: function(encoding) {
          fakeNet.encoding = encoding;
        },
        on: function(event, cb) {
          fakeNet.cbs[event] = cb;
        }
      });

      return {
        listen: function(port, host) {
          fakeNet.port = port;
          fakeNet.host = host;
        }
      };
    }
  };
}

vows.describe('Multiprocess Appender').addBatch({
  'worker': {
    topic: function() {
      var fakeNet = makeFakeNet(),
      appender = sandbox.require(
        '../lib/appenders/multiprocess',
        {
          requires: {
            'net': fakeNet
          }
        }
      ).appender({ mode: 'worker', loggerPort: 1234, loggerHost: 'pants' });
      
      //don't need a proper log event for the worker tests
      appender('before connect');
      fakeNet.cbs.connect();
      appender('after connect');
      fakeNet.cbs.close(true);
      appender('after error, before connect');
      fakeNet.cbs.connect();
      appender('after error, after connect');
	  appender(new Error('Error test'));
      
      return fakeNet;
    },
    'should open a socket to the loggerPort and loggerHost': function(net) {
      assert.equal(net.port, 1234);
      assert.equal(net.host, 'pants');
    },
    'should buffer messages written before socket is connected': function(net) {
      assert.equal(net.data[0], JSON.stringify('before connect'));
    },
    'should write log messages to socket as json strings with a terminator string': function(net) {
      assert.equal(net.data[0], JSON.stringify('before connect'));
      assert.equal(net.data[1], '__LOG4JS__');
      assert.equal(net.data[2], JSON.stringify('after connect'));
      assert.equal(net.data[3], '__LOG4JS__');
      assert.equal(net.encoding, 'utf8');
    },
    'should attempt to re-open the socket on error': function(net) {
      assert.equal(net.data[4], JSON.stringify('after error, before connect'));
      assert.equal(net.data[5], '__LOG4JS__');
      assert.equal(net.data[6], JSON.stringify('after error, after connect'));
      assert.equal(net.data[7], '__LOG4JS__');
      assert.equal(net.createConnectionCalled, 2);
    },
    'should serialize an Error correctly': function(net) {
      assert(JSON.parse(net.data[8]).stack, "Expected:\n\n" + net.data[8] + "\n\n to have a 'stack' property");
      var actual = JSON.parse(net.data[8]).stack;
      var expectedRegex = /^Error: Error test/;
      assert(actual.match(expectedRegex), "Expected: \n\n " + actual + "\n\n to match " + expectedRegex);

    }
  },
  'worker with timeout': {
    topic: function() {
      var fakeNet = makeFakeNet(),
      appender = sandbox.require(
        '../lib/appenders/multiprocess',
        {
          requires: {
            'net': fakeNet
          }
        }
      ).appender({ mode: 'worker' });
      
      //don't need a proper log event for the worker tests
      appender('before connect');
      fakeNet.cbs.connect();
      appender('after connect');
      fakeNet.cbs.timeout();
      appender('after timeout, before close');
      fakeNet.cbs.close();
      appender('after close, before connect');
      fakeNet.cbs.connect();
      appender('after close, after connect');
      
      return fakeNet;
    },
    'should attempt to re-open the socket': function(net) {
      //skipping the __LOG4JS__ separators
      assert.equal(net.data[0], JSON.stringify('before connect'));
      assert.equal(net.data[2], JSON.stringify('after connect'));
      assert.equal(net.data[4], JSON.stringify('after timeout, before close'));
      assert.equal(net.data[6], JSON.stringify('after close, before connect'));
      assert.equal(net.data[8], JSON.stringify('after close, after connect'));
      assert.equal(net.createConnectionCalled, 2);
    }
  },
  'worker defaults': {
    topic: function() {
      var fakeNet = makeFakeNet(),
      appender = sandbox.require(
        '../lib/appenders/multiprocess',
        {
          requires: {
            'net': fakeNet
          }
        }
      ).appender({ mode: 'worker' });
      
      return fakeNet;
    },
    'should open a socket to localhost:5000': function(net) {
      assert.equal(net.port, 5000);
      assert.equal(net.host, 'localhost');
    }
  },
  'master': {
    topic: function() {
      var fakeNet = makeFakeNet(),
      appender = sandbox.require(
        '../lib/appenders/multiprocess',
        {
          requires: {
            'net': fakeNet
          }
        }
      ).appender({ mode: 'master',
                   loggerHost: 'server',
                   loggerPort: 1234,
                   actualAppender: fakeNet.fakeAppender.bind(fakeNet)
                 });
      
      appender('this should be sent to the actual appender directly');
      
      return fakeNet;
    },
    'should listen for log messages on loggerPort and loggerHost': function(net) {
      assert.equal(net.port, 1234);
      assert.equal(net.host, 'server');
    },
    'should return the underlying appender': function(net) {
      assert.equal(net.logEvents[0], 'this should be sent to the actual appender directly');
    },
    'when a client connects': {
      topic: function(net) {
        var logString = JSON.stringify(
          { level: { level: 10000, levelStr: 'DEBUG' }
            , data: ['some debug']}
        ) + '__LOG4JS__';
        
        net.cbs.data(
          JSON.stringify(
            { level: { level: 40000, levelStr: 'ERROR' }
              , data: ['an error message'] }
          ) + '__LOG4JS__'
        );
        net.cbs.data(logString.substring(0, 10));
        net.cbs.data(logString.substring(10));
        net.cbs.data(logString + logString + logString);
        net.cbs.end(
          JSON.stringify(
            { level:  { level: 50000, levelStr: 'FATAL' }
              , data: ["that's all folks"] }
          ) + '__LOG4JS__'
        );
        net.cbs.data('bad message__LOG4JS__');
        return net;
      },
      'should parse log messages into log events and send to appender': function(net) {
        assert.equal(net.logEvents[1].level.toString(), 'ERROR');
        assert.equal(net.logEvents[1].data[0], 'an error message');
        assert.equal(net.logEvents[1].remoteAddress, '1.2.3.4');
        assert.equal(net.logEvents[1].remotePort, '1234');
      },
      'should parse log messages split into multiple chunks': function(net) {
        assert.equal(net.logEvents[2].level.toString(), 'DEBUG');
        assert.equal(net.logEvents[2].data[0], 'some debug');
        assert.equal(net.logEvents[2].remoteAddress, '1.2.3.4');
        assert.equal(net.logEvents[2].remotePort, '1234');
      },
      'should parse multiple log messages in a single chunk': function(net) {
        assert.equal(net.logEvents[3].data[0], 'some debug');
        assert.equal(net.logEvents[4].data[0], 'some debug');
        assert.equal(net.logEvents[5].data[0], 'some debug');
      },
      'should handle log messages sent as part of end event': function(net) {
        assert.equal(net.logEvents[6].data[0], "that's all folks");
      },
      'should handle unparseable log messages': function(net) {
        assert.equal(net.logEvents[7].level.toString(), 'ERROR');
        assert.equal(net.logEvents[7].categoryName, 'log4js');
        assert.equal(net.logEvents[7].data[0], 'Unable to parse log:');
        assert.equal(net.logEvents[7].data[1], 'bad message');
      }
    }
  },
  'master defaults': {
    topic: function() {
      var fakeNet = makeFakeNet(),
      appender = sandbox.require(
        '../lib/appenders/multiprocess',
        {
          requires: {
            'net': fakeNet
          }
        }
      ).appender({ mode: 'master' });
      
      return fakeNet;
    },
    'should listen for log messages on localhost:5000': function(net) {
      assert.equal(net.port, 5000);
      assert.equal(net.host, 'localhost');
    }
  }
}).addBatch({
  'configure': {
    topic: function() {
      var results = {}
      , fakeNet = makeFakeNet()
      , appender = sandbox.require(
        '../lib/appenders/multiprocess',
        {
          requires: {
            'net': fakeNet,
            '../log4js': {
              loadAppender: function(app) {
                results.appenderLoaded = app;
              },
              appenderMakers: {
                'madeupappender': function(config, options) {
                  results.config = config;
                  results.options = options;
                }
              }
            }
          }
        }
      ).configure(
        {
          mode: 'master',
          appender: {
            type: 'madeupappender',
            cheese: 'gouda'
          }
        },
        { crackers: 'jacobs' }
      );

      return results;
        
    },
    'should load underlying appender for master': function(results) {
      assert.equal(results.appenderLoaded, 'madeupappender');
    },
    'should pass config to underlying appender': function(results) {
      assert.equal(results.config.cheese, 'gouda');
    },
    'should pass options to underlying appender': function(results) {
      assert.equal(results.options.crackers, 'jacobs');
    }
  }
}).exportTo(module);
