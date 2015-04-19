"use strict";
var layouts = require('../layouts')
, dgram = require('dgram')
, util = require('util');

function logstashUDP (config, layout) {
  var udp = dgram.createSocket('udp4');
  var type = config.logType ? config.logType : config.category;
  layout = layout || layouts.colouredLayout;
  if(!config.fields) {
    config.fields = {};
  }
  return function(loggingEvent) {
    var logMessage = layout(loggingEvent);
    var fields = {};
    for(var i in config.fields) {
      fields[i] = config.fields[i];
    }
    fields['level'] = loggingEvent.level.levelStr;
    var logObject = {
      '@timestamp': (new Date(loggingEvent.startTime)).toISOString(),
      type: type,
      message: logMessage,
      fields: fields
    };
    sendLog(udp, config.host, config.port, logObject);
  };
}

function sendLog(udp, host, port, logObject) {
  var buffer = new Buffer(JSON.stringify(logObject));
  udp.send(buffer, 0, buffer.length, port, host, function(err, bytes) {
    if(err) {
      console.error(
        "log4js.logstashUDP - %s:%p Error: %s", host, port, util.inspect(err)
      );
    }
  });
}

function configure(config) {
  var layout;
  if (config.layout) {
    layout = layouts.layout(config.layout.type, config.layout);
  }
  return logstashUDP(config, layout);
}

exports.appender = logstashUDP;
exports.configure = configure;
