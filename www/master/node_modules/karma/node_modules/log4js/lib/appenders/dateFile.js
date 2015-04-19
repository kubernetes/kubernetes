"use strict";
var streams = require('../streams')
, layouts = require('../layouts')
, async = require('async')
, path = require('path')
, os = require('os')
, eol = os.EOL || '\n'
, openFiles = [];

//close open files on process exit.
process.on('exit', function() {
  openFiles.forEach(function (file) {
    file.end();
  });
});

/**
 * File appender that rolls files according to a date pattern.
 * @filename base filename.
 * @pattern the format that will be added to the end of filename when rolling,
 *          also used to check when to roll files - defaults to '.yyyy-MM-dd'
 * @layout layout function for log messages - defaults to basicLayout
 * @timezoneOffset optional timezone offset in minutes - defaults to system local
 */
function appender(filename, pattern, alwaysIncludePattern, layout, timezoneOffset) {
  layout = layout || layouts.basicLayout;

  var logFile = new streams.DateRollingFileStream(
    filename,
    pattern,
    { alwaysIncludePattern: alwaysIncludePattern }
  );
  openFiles.push(logFile);

  return function(logEvent) {
    logFile.write(layout(logEvent, timezoneOffset) + eol, "utf8");
  };

}

function configure(config, options) {
  var layout;

  if (config.layout) {
    layout = layouts.layout(config.layout.type, config.layout);
  }

  if (!config.alwaysIncludePattern) {
    config.alwaysIncludePattern = false;
  }

  if (options && options.cwd && !config.absolute) {
    config.filename = path.join(options.cwd, config.filename);
  }

  return appender(config.filename, config.pattern, config.alwaysIncludePattern, layout, config.timezoneOffset);
}

function shutdown(cb) {
  async.each(openFiles, function(file, done) {
    if (!file.write(eol, "utf-8")) {
      file.once('drain', function() {
        file.end(done);
      });
    } else {
      file.end(done);
    }
  }, cb);
}

exports.appender = appender;
exports.configure = configure;
exports.shutdown = shutdown;
