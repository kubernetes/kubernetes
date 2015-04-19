"use strict";
var debug = require('../debug')('fileSync')
, layouts = require('../layouts')
, path = require('path')
, fs = require('fs')
, streams = require('../streams')
, os = require('os')
, eol = os.EOL || '\n'
;

function RollingFileSync (filename, size, backups, options) {
  debug("In RollingFileStream");

  function throwErrorIfArgumentsAreNotValid() {
    if (!filename || !size || size <= 0) {
      throw new Error("You must specify a filename and file size");
    }
  }
  
  throwErrorIfArgumentsAreNotValid();
  
  this.filename = filename;
  this.size = size;
  this.backups = backups || 1;
  this.options = options || { encoding: 'utf8', mode: parseInt('0644', 8), flags: 'a' };
  this.currentSize = 0;
  
  function currentFileSize(file) {
    var fileSize = 0;
    try {
      fileSize = fs.statSync(file).size;
    } catch (e) {
      // file does not exist
      fs.appendFileSync(filename, '');
    }
    return fileSize;
  }

  this.currentSize = currentFileSize(this.filename);
}

RollingFileSync.prototype.shouldRoll = function() {
  debug("should roll with current size %d, and max size %d", this.currentSize, this.size);
  return this.currentSize >= this.size;
};

RollingFileSync.prototype.roll = function(filename) {
  var that = this,
  nameMatcher = new RegExp('^' + path.basename(filename));
  
  function justTheseFiles (item) {
    return nameMatcher.test(item);
  }
  
  function index(filename_) {
    return parseInt(filename_.substring((path.basename(filename) + '.').length), 10) || 0;
  }
  
  function byIndex(a, b) {
    if (index(a) > index(b)) {
      return 1;
    } else if (index(a) < index(b) ) {
      return -1;
    } else {
      return 0;
    }
  }

  function increaseFileIndex (fileToRename) {
    var idx = index(fileToRename);
    debug('Index of ' + fileToRename + ' is ' + idx);
    if (idx < that.backups) {
      //on windows, you can get a EEXIST error if you rename a file to an existing file
      //so, we'll try to delete the file we're renaming to first
      try {
        fs.unlinkSync(filename + '.' + (idx+1));
      } catch(e) {
        //ignore err: if we could not delete, it's most likely that it doesn't exist
      }
      
      debug('Renaming ' + fileToRename + ' -> ' + filename + '.' + (idx+1));
      fs.renameSync(path.join(path.dirname(filename), fileToRename), filename + '.' + (idx + 1));
    }
  }

  function renameTheFiles() {
    //roll the backups (rename file.n to file.n+1, where n <= numBackups)
    debug("Renaming the old files");
    
    var files = fs.readdirSync(path.dirname(filename));
    files.filter(justTheseFiles).sort(byIndex).reverse().forEach(increaseFileIndex);
  }

  debug("Rolling, rolling, rolling");
  renameTheFiles();
};

RollingFileSync.prototype.write = function(chunk, encoding) {
  var that = this;
  
  
  function writeTheChunk() {
    debug("writing the chunk to the file");
    that.currentSize += chunk.length;
    fs.appendFileSync(that.filename, chunk);
  }

  debug("in write");
  

  if (this.shouldRoll()) {
    this.currentSize = 0;
    this.roll(this.filename);
  }
  
  writeTheChunk();
};


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
 * @param timezoneOffset - optional timezone offset in minutes
 *   (default system local)
 */
function fileAppender (file, layout, logSize, numBackups, timezoneOffset) {
  debug("fileSync appender created");
  var bytesWritten = 0;
  file = path.normalize(file);
  layout = layout || layouts.basicLayout;
  numBackups = numBackups === undefined ? 5 : numBackups;
  //there has to be at least one backup if logSize has been specified
  numBackups = numBackups === 0 ? 1 : numBackups;

  function openTheStream(file, fileSize, numFiles) {
    var stream;
    
    if (fileSize) {
      stream = new RollingFileSync(
        file,
        fileSize,
        numFiles
      );
    } else {
      stream = (function(f) {
        // create file if it doesn't exist
        if (!fs.existsSync(f))
            fs.appendFileSync(f, '');
        
        return {
            write: function(data) {
                fs.appendFileSync(f, data);
            }
        };
      })(file);
    }

    return stream;
  }

  var logFile = openTheStream(file, logSize, numBackups);
  
  return function(loggingEvent) {
    logFile.write(layout(loggingEvent, timezoneOffset) + eol);
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

  return fileAppender(config.filename, layout, config.maxLogSize, config.backups, config.timezoneOffset);
}

exports.appender = fileAppender;
exports.configure = configure;
