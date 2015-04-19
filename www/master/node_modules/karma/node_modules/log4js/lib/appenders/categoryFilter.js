"use strict";
var log4js = require('../log4js');

function categoryFilter (excludes, appender) {
  if (typeof(excludes) === 'string') excludes = [excludes];
  return function(logEvent) {
    if (excludes.indexOf(logEvent.categoryName) === -1) {
      appender(logEvent);
    }
  };
}

function configure(config) {
  log4js.loadAppender(config.appender.type);
  var appender = log4js.appenderMakers[config.appender.type](config.appender);
  return categoryFilter(config.exclude, appender);
}

exports.appender = categoryFilter;
exports.configure = configure;
