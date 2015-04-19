/**
 * The launcher is responsible for parsing the capabilities from the
 * input configuration and launching test runners.
 */
'use strict';

var ConfigParser = require('./configParser'),
    TaskScheduler = require('./taskScheduler'),
    helper = require('./util'),
    log = require('./logger'),
    q = require('q'),
    TaskRunner = require('./taskRunner');

var logPrefix = '[launcher]';
var RUNNERS_FAILED_EXIT_CODE = 100;

/**
 * Custom console.log proxy
 * @param {*} [stuff...] Any value to log
 * @private
 */
var log_ = function() {
  var args = [logPrefix].concat([].slice.call(arguments));
  log.puts.apply(log, args);
};

/**
 * Keeps track of a list of task results. Provides method to add a new
 * result, aggregate the results into a summary, count failures,
 * and save results into a JSON file.
 */
var taskResults_ = {
  results_: [],

  add: function(result) {
    this.results_.push(result);
  },

  totalSpecFailures: function() {
    var specFailures = 0;
    this.results_.forEach(function(result) {
      specFailures += result.failedCount;
    });
    return specFailures;
  },

  totalProcessFailures: function() {
    var processFailures = 0;
    this.results_.forEach(function(result) {
      if (!result.failedCount && result.exitCode !== 0) {
        processFailures += 1;
      }
    });
    return processFailures;
  },

  saveResults: function(filepath) {
    var jsonOutput = [];
    this.results_.forEach(function(result) {
      jsonOutput = jsonOutput.concat(result.specResults);
    });

    var json = JSON.stringify(jsonOutput, null, '  ');
    var fs = require('fs');
    fs.writeFileSync(filepath, json);
  },

  reportSummary: function() {
    var specFailures = this.totalSpecFailures();
    var processFailures = this.totalProcessFailures();
    this.results_.forEach(function(result) {
      var capabilities = result.capabilities;
      var shortName = (capabilities.browserName) ? capabilities.browserName : '';
      shortName += (capabilities.version) ? capabilities.version : '';
      shortName += (' #' + result.taskId);
      if (result.failedCount) {
        log_(shortName + ' failed ' + result.failedCount + ' test(s)');
      } else if (result.exitCode !== 0) {
        log_(shortName + ' failed with exit code: ' + result.exitCode);
      } else {
        log_(shortName + ' passed');
      }
    });

    if (specFailures && processFailures) {
      log_('overall: ' + specFailures + ' failed spec(s) and ' +
          processFailures + ' process(es) failed to complete');
    } else if (specFailures) {
      log_('overall: ' + specFailures + ' failed spec(s)');
    } else if (processFailures) {
      log_('overall: ' + processFailures + ' process(es) failed to complete');
    }
  }
};

/**
 * Initialize and run the tests.
 * Exits with 1 on test failure, and RUNNERS_FAILED_EXIT_CODE on unexpected
 * failures.
 *
 * @param {string=} configFile
 * @param {Object=} additionalConfig
 */
var init = function(configFile, additionalConfig) {
  var configParser = new ConfigParser();
  if (configFile) {
    configParser.addFileConfig(configFile);
  }
  if (additionalConfig) {
    configParser.addConfig(additionalConfig);
  }
  var config = configParser.getConfig();
  log.set(config);
  log.debug('Running with --troubleshoot');
  log.debug('Protractor version: ' + require('../package.json').version);
  log.debug('Your base url for tests is ' + config.baseUrl);

  // Run beforeLaunch
  helper.runFilenameOrFn_(config.configDir, config.beforeLaunch).then(function() {

    return q.promise(function(resolve) {
      // 1) If getMultiCapabilities is set, resolve that as `multiCapabilities`.
      if (config.getMultiCapabilities &&
          typeof config.getMultiCapabilities === 'function') {
        if (config.multiCapabilities.length || config.capabilities) {
          log.warn('getMultiCapabilities() will override both capabilites ' +
                  'and multiCapabilities');
        }
        // If getMultiCapabilities is defined and a function, use this.
        q.when(config.getMultiCapabilities(), function(multiCapabilities) {
          config.multiCapabilities = multiCapabilities;
          config.capabilities = null;
        }).then(resolve);
      } else {
        resolve();
      }
    }).then(function() {
      // 2) Set `multicapabilities` using `capabilities`, `multicapabilites`,
      // or default
      if (config.capabilities) {
        if (config.multiCapabilities.length) {
          log.warn('You have specified both capabilites and ' +
              'multiCapabilities. This will result in capabilities being ' +
              'ignored');
        } else {
          // Use capabilities if multiCapabilities is empty.
          config.multiCapabilities = [config.capabilities];
        }
      } else if (!config.multiCapabilities.length) {
        // Default to chrome if no capabilities given
        config.multiCapabilities = [{
          browserName: 'chrome'
        }];
      }
    });
  }).then(function() {
    // 3) If we're in `elementExplorer` mode, run only that.
    if (config.elementExplorer || config.framework === 'explorer') {
      if (config.multiCapabilities.length != 1) {
        throw new Error('Must specify only 1 browser while using elementExplorer');
      } else {
        config.capabilities = config.multiCapabilities[0];
      }
      config.framework = 'explorer';

      var Runner = require('./runner');
      var runner = new Runner(config);
      return runner.run().then(function(exitCode) {
        process.exit(exitCode);
      }, function(err) {
        log_(err);
        process.exit(1);
      });
    }
  }).then(function() {
    // 4) Run tests.
    var scheduler = new TaskScheduler(config);

    process.on('exit', function(code) {
      if (code) {
        log_('Process exited with error code ' + code);
      } else if (scheduler.numTasksOutstanding() > 0) {
        log_('BUG: launcher exited with ' +
          scheduler.numTasksOutstanding() + ' tasks remaining');
        process.exit(RUNNERS_FAILED_EXIT_CODE);
      }
    });

    // Run afterlaunch and exit
    var cleanUpAndExit = function(exitCode) {
      return helper.runFilenameOrFn_(
          config.configDir, config.afterLaunch, [exitCode]).
            then(function(returned) {
              if (typeof returned === 'number') {
                process.exit(returned);
              } else {
                process.exit(exitCode);
              }
            }, function(err) {
              log_('Error:', err);
              process.exit(1);
            });
    };

    var totalTasks = scheduler.numTasksOutstanding();
    var forkProcess = false;
    if (totalTasks > 1) { // Start new processes only if there are >1 tasks.
      forkProcess = true;
      if (config.debug) {
        throw new Error('Cannot run in debug mode with ' +
          'multiCapabilities, count > 1, or sharding');
      }
    }

    var deferred = q.defer(); // Resolved when all tasks are completed
    var createNextTaskRunner = function() {
      var task = scheduler.nextTask();
      if (task) {
        var taskRunner = new TaskRunner(configFile, additionalConfig, task, forkProcess);
        taskRunner.run().then(function(result) {
          if (result.exitCode && !result.failedCount) {
            log_('Runner process exited unexpectedly with error code: ' + result.exitCode);
          }
          taskResults_.add(result);
          task.done();
          createNextTaskRunner();
          // If all tasks are finished
          if (scheduler.numTasksOutstanding() === 0) {
            deferred.fulfill();
          }
          log_(scheduler.countActiveTasks() +
            ' instance(s) of WebDriver still running');
        }).catch (function(err) {
          log_('Error:', err.stack || err.message || err);
          cleanUpAndExit(RUNNERS_FAILED_EXIT_CODE);
        });
      }
    };
    // Start `scheduler.maxConcurrentTasks()` workers for handling tasks in
    // the beginning. As a worker finishes a task, it will pick up the next task
    // from the scheduler's queue until all tasks are gone.
    for (var i = 0; i < scheduler.maxConcurrentTasks(); ++i) {
      createNextTaskRunner();
    }
    log_('Running ' + scheduler.countActiveTasks() + ' instances of WebDriver');

    // By now all runners have completed.
    deferred.promise.then(function() {
      // Save results if desired
      if (config.resultJsonOutputFile) {
        taskResults_.saveResults(config.resultJsonOutputFile);
      }

      taskResults_.reportSummary();
      var exitCode = 0;
      if (taskResults_.totalProcessFailures() > 0) {
        exitCode = RUNNERS_FAILED_EXIT_CODE;
      } else if (taskResults_.totalSpecFailures() > 0) {
        exitCode = 1;
      }
      return cleanUpAndExit(exitCode);
    }).done();
  }).done();
};

exports.init = init;
