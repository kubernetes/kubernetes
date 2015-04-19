var ConfigParser = require('../configParser'),
  q = require('q');

/**
 * Execute the Runner's test cases through Cucumber.
 *
 * @param {Runner} runner The current Protractor Runner.
 * @param {Array} specs Array of Directory Path Strings.
 * @return {q.Promise} Promise resolved with the test results
 */
exports.run = function(runner, specs) {
  // TODO - add the event interface for cucumber.
  var Cucumber = require('cucumber'),
      execOptions = ['node', 'node_modules/.bin/cucumber-js'],
      cucumberResolvedRequire;

  // Set up exec options for Cucumber
  execOptions = execOptions.concat(specs);
  if (runner.getConfig().cucumberOpts) {

    // Process Cucumber Require param
    if (runner.getConfig().cucumberOpts.require) {
      // TODO - this should move into the launcher.
      cucumberResolvedRequire =
          ConfigParser.resolveFilePatterns(
              runner.getConfig().cucumberOpts.require,
              false,
              runner.getConfig().configDir);
      if (cucumberResolvedRequire && cucumberResolvedRequire.length) {
          execOptions = cucumberResolvedRequire.reduce(function(a, fn) {
            return a.concat('-r', fn);
          }, execOptions);
      }
    }

    // Process Cucumber Tag param
    if (Array.isArray(runner.getConfig().cucumberOpts.tags)) {
        for (var i in runner.getConfig().cucumberOpts.tags) {
            var tags = runner.getConfig().cucumberOpts.tags[i];
            execOptions.push('-t');
            execOptions.push(tags);
        }
    } else if (runner.getConfig().cucumberOpts.tags) {
      execOptions.push('-t');
      execOptions.push(runner.getConfig().cucumberOpts.tags);
    }

    // Process Cucumber Format param
    if (runner.getConfig().cucumberOpts.format) {
      execOptions.push('-f');
      execOptions.push(runner.getConfig().cucumberOpts.format);
    }

    // Process Cucumber 'coffee' param
    if (runner.getConfig().cucumberOpts.coffee) {
      execOptions.push('--coffee');
    }
  }
  global.cucumber = Cucumber.Cli(execOptions);

  var testResult = [];
  var stepResults = {
    description: null,
    assertions: [],
    duration: 0
  };
  var scenarioFailed = false;

  var failedCount = 0;
  // Add a listener into cucumber so that protractor can find out which
  // steps passed/failed
  var addResultListener = function(formatter) {
    var originalHandleAfterScenarioEvent = formatter.handleAfterScenarioEvent;
    formatter.handleAfterScenarioEvent = function(event, callback) {
      var scenario = event.getPayloadItem('scenario');
      stepResults.description = scenario.getName();
      if (scenarioFailed) {
        ++failedCount;
        runner.emit('testFail');
      } else {
        runner.emit('testPass');
      }

      testResult.push(stepResults);
      stepResults = {
        description: null,
        assertions: [],
        duration: 0
      };
      scenarioFailed = false;

      if (originalHandleAfterScenarioEvent
          && typeof(originalHandleAfterScenarioEvent) === 'function') {
            originalHandleAfterScenarioEvent(event, callback);
      } else {
        callback();
      }
    };

    var originalHandleStepResultEvent = formatter.handleStepResultEvent;
    formatter.handleStepResultEvent = function(event, callback) {
      var stepResult = event.getPayloadItem('stepResult');
      if (stepResult.isSuccessful()) {
        stepResults.assertions.push({
          passed: true
        });
        stepResults.duration += stepResult.getDuration();
      } else if (stepResult.isFailed()) {
        scenarioFailed = true;
        var failureMessage = stepResult.getFailureException();
        stepResults.assertions.push({
          passed: false,
          errorMsg: failureMessage.message,
          stackTrace: failureMessage.stack
        });
        stepResults.duration += stepResult.getDuration();
      }

      if (originalHandleStepResultEvent
          && typeof(originalHandleStepResultEvent) === 'function') {
            originalHandleStepResultEvent(event, callback);
      } else {
        callback();
      }
    };
  };

  return runner.runTestPreparer().then(function() {
    return q.promise(function(resolve, reject) {
      var cucumberConf = Cucumber.Cli.Configuration(execOptions);
      var runtime = Cucumber.Runtime(cucumberConf);
      var formatter = cucumberConf.getFormatter();
      addResultListener(formatter);
      runtime.attachListener(formatter);
      runtime.start(function() {
        try {
          if (runner.getConfig().onComplete) {
            runner.getConfig().onComplete();
          }
          resolve({
            failedCount: failedCount,
            specResults: testResult
          });
        } catch (err) {
          reject(err);
        }
      });
    });
  });
};
