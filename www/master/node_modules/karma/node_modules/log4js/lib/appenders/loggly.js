'use strict';
var layouts = require('../layouts')
, loggly = require('loggly')
, os = require('os')
, passThrough = layouts.messagePassThroughLayout;


/**
 * Loggly Appender. Sends logging events to Loggly using node-loggly 
 *
 * @param config object with loggly configuration data
 * {
 *   token: 'your-really-long-input-token',
 *   subdomain: 'your-subdomain',
 *   tags: ['loggly-tag1', 'loggly-tag2', .., 'loggly-tagn'] 
 * }
 * @param layout a function that takes a logevent and returns a string (defaults to objectLayout).
 */
function logglyAppender(config, layout) {
	var client = loggly.createClient(config);
  if(!layout) layout = passThrough;

  return function(loggingEvent) {
		var msg = layout(loggingEvent);
		client.log({
			msg: msg,
			level: loggingEvent.level.levelStr,
			category: loggingEvent.categoryName,
			hostname: os.hostname().toString(),
		});
  }
}

function configure(config) {
	var layout;
	if (config.layout) {
		layout = layouts.layout(config.layout.type, config.layout);
	}
	return logglyAppender(config, layout);
}

exports.name      = 'loggly';
exports.appender  = logglyAppender;
exports.configure = configure;
