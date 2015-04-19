var EOL = require('os').EOL;
var log = require('./logger.js');

/**
 * Log output such that metadata are appended.
 * Calling log(data) will not flush to console until you call flush()
 *
 * @constructor
 * @param {object} task Task that is being reported.
 * @param {number} pid PID of process running the task.
 */
var TaskLogger = function(task, pid) {
  this.task = task;
  this.pid = pid;
  this.buffer = '';
  this.insertTag = true;

  this.logHeader_();
};

/**
 * Log the header for the current task including information such as
 * PID, browser name/version, task Id, specs being run.
 *
 * @private
 */
TaskLogger.prototype.logHeader_ = function() {
  var output = 'PID: ' + this.pid + EOL;
  if (this.task.specs.length === 1) {
    output += 'Specs: ' + this.task.specs.toString() + EOL + EOL;
  }
  this.log(output);
};


/**
 * Flushes the buffer to stdout.
 */
TaskLogger.prototype.flush = function() {
  if (this.buffer) {
    // Flush buffer if nonempty
    log.print(EOL + '------------------------------------' + EOL);
    log.print(this.buffer);
    log.print(EOL);
    this.buffer = '';
  }
};

/**
 * Log the data in the argument such that metadata are appended.
 * The data will be saved to a buffer until flush() is called.
 *
 * @param {string} data
 */
TaskLogger.prototype.log = function(data) {
  var tag = '[';
  var capabilities = this.task.capabilities;
  tag += (capabilities.browserName) ?
    capabilities.browserName : '';
  tag += (capabilities.version) ?
    (' ' + capabilities.version) : '';
  tag += (capabilities.platform) ?
    (' ' + capabilities.platform) : '';
  tag += (' #' + this.task.taskId);
  tag += '] ';

  data = data.toString();
  for (var i = 0; i < data.length; i++) {
    if (this.insertTag) {
      this.insertTag = false;
      // This ensures that the '\x1B[0m' appears before the tag, so that
      // data remains correct when color is not processed.
      // See https://github.com/angular/protractor/pull/1216
      if (data[i] === '\x1B' && data.substring(i, i + 4) === '\x1B[0m') {
        this.buffer += ('\x1B[0m' + tag);
        i += 3;
        continue;
      }

      this.buffer += tag;
    }
    if (data[i] === '\n') {
      this.insertTag = true;
    }
    this.buffer += data[i];
  }
};

module.exports = TaskLogger;
