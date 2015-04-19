var child = require('child_process');
var q = require('q');
var TaskLogger = require('./taskLogger.js');
var EventEmitter = require('events').EventEmitter;
var util = require('util');
var log = require('./logger.js');


/**
 * A runner for running a specified task (capabilities + specs).
 * The TaskRunner can either run the task from the current process (via
 * './runner.js') or from a new process (via './runnerCli.js').
 *
 * @constructor
 * @param {string} configFile Path of test configuration.
 * @param {object} additionalConfig Additional configuration.
 * @param {object} task Task to run.
 * @param {boolean} runInFork Whether to run test in a forked process.
 * @constructor
 */
var TaskRunner = function(configFile, additionalConfig, task, runInFork) {
  this.configFile = configFile;
  this.additionalConfig = additionalConfig;
  this.task = task;
  this.runInFork = runInFork;
};
util.inherits(TaskRunner, EventEmitter);

/**
 * Sends the run command.
 * @return {q.Promise} A promise that will resolve when the task finishes
 *     running. The promise contains the following parameters representing the
 *     result of the run:
 *       taskId, specs, capabilities, failedCount, exitCode, specResults
 */
TaskRunner.prototype.run = function() {
  var runResults = {
    taskId: this.task.taskId,
    specs: this.task.specs,
    capabilities: this.task.capabilities,
    // The following are populated while running the test:
    failedCount: 0,
    exitCode: -1,
    specResults: []
  };

  if (this.runInFork) {
    var deferred = q.defer();

    var childProcess = child.fork(
      __dirname + '/runnerCli.js',
      process.argv.slice(2), {
        cwd: process.cwd(),
        silent: true
      }
    );
    var taskLogger = new TaskLogger(this.task, childProcess.pid);

    // stdout pipe
    childProcess.stdout.on('data', function(data) {
      taskLogger.log(data);
    });

    // stderr pipe
    childProcess.stderr.on('data', function(data) {
      taskLogger.log(data);
    });

    childProcess.on('message', function(m) {
      switch (m.event) {
        case 'testPass':
          log.print('.');
          break;
        case 'testFail':
          log.print('F');
          break;
        case 'testsDone':
          runResults.failedCount = m.results.failedCount;
          runResults.specResults = m.results.specResults;
          break;
      }
    })
    .on('error', function(err) {
      taskLogger.flush();
      deferred.reject(err);
    })
    .on('exit', function(code) {
      taskLogger.flush();
      runResults.exitCode = code;
      deferred.resolve(runResults);
    });

    childProcess.send({
      command: 'run',
      configFile: this.configFile,
      additionalConfig: this.additionalConfig,
      capabilities: this.task.capabilities,
      specs: this.task.specs
    });

    return deferred.promise;
  } else {
    var ConfigParser = require('./configParser');
    var configParser = new ConfigParser();
    if (this.configFile) {
      configParser.addFileConfig(this.configFile);
    }
    if (this.additionalConfig) {
      configParser.addConfig(this.additionalConfig);
    }
    var config = configParser.getConfig();
    config.capabilities = this.task.capabilities;
    config.specs = this.task.specs;

    var Runner = require('./runner');
    var runner = new Runner(config);

    runner.on('testsDone', function(results) {
      runResults.failedCount = results.failedCount;
      runResults.specResults = results.specResults;
    });

    return runner.run().then(function(exitCode) {
      runResults.exitCode = exitCode;
      return runResults;
    });
  }
};

module.exports = TaskRunner;
