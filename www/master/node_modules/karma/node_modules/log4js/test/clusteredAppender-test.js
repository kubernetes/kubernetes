"use strict";
var assert = require('assert');
var vows = require('vows');
var layouts = require('../lib/layouts');
var sandbox = require('sandboxed-module');
var LoggingEvent = require('../lib/logger').LoggingEvent;
var cluster = require('cluster');

vows.describe('log4js cluster appender').addBatch({
	'when in master mode': {
		topic: function() {

			var registeredClusterEvents = [];
			var loggingEvents = [];
			
			// Fake cluster module, so no cluster listeners be really added 
			var fakeCluster = {
			
				on: function(event, callback) {
					registeredClusterEvents.push(event);
				},
				
				isMaster: true,
				isWorker: false,
				
			};
		
			var fakeActualAppender = function(loggingEvent) {
				loggingEvents.push(loggingEvent);
			}
			
			// Load appender and fake modules in it
			var appenderModule = sandbox.require('../lib/appenders/clustered', {
				requires: {
					'cluster': fakeCluster,
				}
			});
		
			var masterAppender = appenderModule.appender({
				actualAppenders: [fakeActualAppender, fakeActualAppender, fakeActualAppender],
                                appenders: [{}, {category: "test"}, {category: "wovs"}]
			});

			// Actual test - log message using masterAppender
			masterAppender(new LoggingEvent('wovs', 'Info', ['masterAppender test']));
			
			var returnValue = {
				registeredClusterEvents: registeredClusterEvents,
				loggingEvents: loggingEvents,
			};
		
			return returnValue;
		}, 
		
		"should register 'fork' event listener on 'cluster'": function(topic) { 
			assert.equal(topic.registeredClusterEvents[0], 'fork');
		},
		
		"should log using actual appender": function(topic) {
                        assert.equal(topic.loggingEvents.length, 2)
			assert.equal(topic.loggingEvents[0].data[0], 'masterAppender test');
                        assert.equal(topic.loggingEvents[1].data[0], 'masterAppender test');
		},
		
	},
	
	'when in worker mode': {
		
		topic: function() {
			
			var registeredProcessEvents = [];
			
			// Fake cluster module, to fake we're inside a worker process
			var fakeCluster = {
			
				isMaster: false,
				isWorker: true,
				
			};
			
			var fakeProcess = {
			
				send: function(data) {
					registeredProcessEvents.push(data);
				},
				
			};
			
			// Load appender and fake modules in it
			var appenderModule = sandbox.require('../lib/appenders/clustered', {
				requires: {
					'cluster': fakeCluster,
				},
				globals: {
					'process': fakeProcess,
				}
			});
			
			var workerAppender = appenderModule.appender();

			// Actual test - log message using masterAppender
			workerAppender(new LoggingEvent('wovs', 'Info', ['workerAppender test']));
			workerAppender(new LoggingEvent('wovs', 'Info', [new Error('Error test')]));
			
			var returnValue = {
				registeredProcessEvents: registeredProcessEvents,
			};
			
			return returnValue;
		
		},
		
		"worker appender should call process.send" : function(topic) {
			assert.equal(topic.registeredProcessEvents[0].type, '::log-message');
			assert.equal(JSON.parse(topic.registeredProcessEvents[0].event).data[0], "workerAppender test");
		},
		
		"worker should serialize an Error correctly" : function(topic) {
			assert.equal(topic.registeredProcessEvents[1].type, '::log-message');
			assert(JSON.parse(topic.registeredProcessEvents[1].event).data[0].stack);
			var actual = JSON.parse(topic.registeredProcessEvents[1].event).data[0].stack;
			var expectedRegex = /^Error: Error test/;
			assert(actual.match(expectedRegex), "Expected: \n\n " + actual + "\n\n to match " + expectedRegex);
		}
		
	}

}).exportTo(module);
