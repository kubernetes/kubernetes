"use strict";
var levels = require('../levels')
, log4js = require('../log4js');

function logLevelFilter (minLevelString, maxLevelString, appender) {
  var minLevel = levels.toLevel(minLevelString);
  var maxLevel = levels.toLevel(maxLevelString, levels.FATAL);
  return function(logEvent) {
      var eventLevel = logEvent.level;
      if (eventLevel.isGreaterThanOrEqualTo(minLevel) && eventLevel.isLessThanOrEqualTo(maxLevel)) {
      appender(logEvent);
    }
  };
}

function configure(config, options) {
  log4js.loadAppender(config.appender.type);
  var appender = log4js.appenderMakers[config.appender.type](config.appender, options);
  return logLevelFilter(config.level, config.maxLevel, appender);
}

exports.appender = logLevelFilter;
exports.configure = configure;
