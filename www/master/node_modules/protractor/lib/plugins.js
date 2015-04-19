var q = require('q'),
    helper = require('./util'),
    ConfigParser = require('./configParser'),
    log = require('./logger');

var logPrefix = '[plugins]';

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
 * The plugin API for Protractor.  Note that this API is unstable. See
 * plugins/README.md for more information.
 *
 * @constructor
 * @param {Object} config parsed from the config file
 */
var Plugins = function(config) {
  var self = this;

  this.pluginConfs = config.plugins || [];
  this.pluginObjs = [];
  this.pluginConfs.forEach(function(pluginConf) {
    var path;
    if (pluginConf.path) {
      path = ConfigParser.resolveFilePatterns(pluginConf.path, true,
          config.configDir)[0];
    } else {
      path = pluginConf.package;
    }
    if (!path) {
      throw new Error('Plugin configuration did not contain a valid path.');
    }
    pluginConf.name = path;
    self.pluginObjs.push(require(path));
  });
};

var noop = function() {};

Plugins.printPluginResults = function(specResults) {
  var green = '\x1b[32m';
  var red = '\x1b[31m';
  var normalColor = '\x1b[39m';

  var printResult = function(message, pass) {
    log.puts(pass ? green : red,
        '\t', pass ? 'Pass: ' : 'Fail: ', message, normalColor);
  };

  for (var j = 0; j < specResults.length; j++) {
    var specResult = specResults[j];
    var passed = specResult.assertions.map(function(x) {
      return x.passed;
    }).reduce(function(x, y) {
      return x && y;
    }, true);

    printResult(specResult.description, passed);
    if (!passed) {
      for (var k = 0; k < specResult.assertions.length; k++) {
        if (!specResult.assertions[k].passed) {
          log.puts('\t\t' + specResult.assertions[k].errorMsg);
        }
      }
    }
  }
};

function pluginFunFactory(funName) {
  return function() {
    var names = [];
    var promises = [];
    for (var i = 0; i < this.pluginConfs.length; ++i) {
      var pluginConf = this.pluginConfs[i];
      var pluginObj = this.pluginObjs[i];
      names.push(pluginObj.name || pluginConf.name);
      promises.push(
          (pluginObj[funName] || noop).apply(
              pluginObj[funName],
              [pluginConf].concat([].slice.call(arguments))));
    }

    return q.all(promises).then(function(results) {
      // Join the results into a single object and output any test results
      var ret = {failedCount: 0};

      for (var i = 0; i < results.length; i++) {
        var pluginResult = results[i];
        if (!!pluginResult && (typeof pluginResult == typeof {})) {
          if (typeof pluginResult.failedCount != typeof 1) {
            log_('Plugin "' + names[i] + '" returned a malformed object');
            continue; // Just ignore this result
          }

          // Output test results
          if (pluginResult.specResults) {
            log.puts('Plugin: ' + names[i] + ' (' + funName + ')');
            Plugins.printPluginResults(pluginResult.specResults);
          }

          // Join objects
          ret = helper.joinTestLogs(ret, pluginResult);
        }
      }

      return ret;
    });
  };
}

/**
 * Sets up plugins before tests are run.
 *
 * @return {q.Promise} A promise which resolves when the plugins have all been
 *     set up.
 */
Plugins.prototype.setup = pluginFunFactory('setup');

/**
 * Tears down plugins after tests are run.
 *
 * @return {q.Promise} A promise which resolves when the plugins have all been
 *     torn down.
 */
Plugins.prototype.teardown = pluginFunFactory('teardown');

/**
 * Run after the test results have been processed (any values returned will
 * be ignored), but before the process exits. Final chance for cleanup.
 *
 * @return {q.Promise} A promise which resolves when the plugins have all been
 *     torn down.
 */
Plugins.prototype.postResults = pluginFunFactory('postResults');

/**
 * Called after each test block completes.
 *
 * @return {q.Promise} A promise which resolves when the plugins have all been
 *     torn down.
 */
Plugins.prototype.postTest = pluginFunFactory('postTest');

module.exports = Plugins;
