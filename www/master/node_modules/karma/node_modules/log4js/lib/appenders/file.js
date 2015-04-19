"use strict";
var layouts = require('../layouts')
, async = require('async')
, path = require('path')
, fs = require('fs')
, streams = require('../streams')
, os = require('os')
, eol = os.EOL || '\n'
, openFiles = []
, levels = require('../levels');

//close open files on process exit.
process.on('exit', function() {
  openFiles.forEach(function (file) {
    file.end();
  });
});

/**
 * File Appender writing the logs to a text file. Supports rolling of logs by size.
 *
 * @param file file log messages will be written to
 * @param layout a function that takes a logevent and returns a string 
 *   (defaults to basicLayout).
 * @param logSize - the maximum size (in bytes) for a log file, 
 *   if not provided then logs won't be rotated.
 * @param numBackups - the number of log files to keep after logSize 
 *   has been reached (default 5)
 * @param compress - flag that controls log file compression
 * @param timezoneOffset - optional timezone offset in minutes (default system local)
 */
function fileAppender (file, layout, logSize, numBackups, compress, timezoneOffset) {
  var bytesWritten = 0;
  file = path.normalize(file);
  layout = layout || layouts.basicLayout;
  numBackups = numBackups === undefined ? 5 : numBackups;
  //there has to be at least one backup if logSize has been specified
  numBackups = numBackups === 0 ? 1 : numBackups;

  function openTheStream(file, fileSize, numFiles) {
    var stream;
    if (fileSize) {
      stream = new streams.RollingFileStream(
        file,
        fileSize,
        numFiles,
        { "compress": compress }
      );
    } else {
      stream = fs.createWriteStream(
        file, 
        { encoding: "utf8", 
          mode: parseInt('0644', 8), 
          flags: 'a' }
      );
    }
    stream.on("error", function (err) {
      console.error("log4js.fileAppender - Writing to file %s, error happened ", file, err);
    });
    return stream;
  }

  var logFile = openTheStream(file, logSize, numBackups);
  
  // push file to the stack of open handlers
  openFiles.push(logFile);

  return function(loggingEvent) {
    logFile.write(layout(loggingEvent, timezoneOffset) + eol, "utf8");
  };

}

function configure(config, options) {
  var layout;
  if (config.layout) {
    layout = layouts.layout(config.layout.type, config.layout);
  }

  if (options && options.cwd && !config.absolute) {
    config.filename = path.join(options.cwd, config.filename);
  }

  return fileAppender(config.filename, layout, config.maxLogSize, config.backups, config.compress, config.timezoneOffset);
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

exports.appender = fileAppender;
exports.configure = configure;
exports.shutdown = shutdown;
