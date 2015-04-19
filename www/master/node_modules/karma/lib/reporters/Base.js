var util = require('util');

var helper = require('../helper');


var BaseReporter = function(formatError, reportSlow, adapter) {
  this.adapters = [adapter || process.stdout.write.bind(process.stdout)];

  this.onRunStart = function(browsers) {
    this._browsers = browsers;
  };


  this.renderBrowser = function(browser) {
    var results = browser.lastResult;
    var totalExecuted = results.success + results.failed;
    var msg = util.format('%s: Executed %d of %d', browser, totalExecuted, results.total);

    if (results.failed) {
      msg += util.format(this.X_FAILED, results.failed);
    }

    if (results.skipped) {
      msg += util.format(' (skipped %d)', results.skipped);
    }

    if (browser.isReady) {
      if (results.disconnected) {
        msg += this.FINISHED_DISCONNECTED;
      } else if (results.error) {
        msg += this.FINISHED_ERROR;
      } else if (!results.failed) {
        msg += this.FINISHED_SUCCESS;
      }

      msg += util.format(' (%s / %s)', helper.formatTimeInterval(results.totalTime),
                                       helper.formatTimeInterval(results.netTime));
    }

    return msg;
  };

  this.renderBrowser = this.renderBrowser.bind(this);


  this.write = function() {
    var msg = util.format.apply(null, Array.prototype.slice.call(arguments));

    this.adapters.forEach(function(adapter) {
      adapter(msg);
    });
  };

  this.writeCommonMsg = this.write;


  this.onBrowserError = function(browser, error) {
    this.writeCommonMsg(util.format(this.ERROR, browser) + formatError(error, '\t'));
  };


  this.onBrowserLog = function(browser, log, type) {
    if (!helper.isString(log)) {
      // TODO(vojta): change util to new syntax (config object)
      log = util.inspect(log, false, undefined, this.USE_COLORS);
    }

    if (this._browsers && this._browsers.length === 1) {
      this.writeCommonMsg(util.format(this.LOG_SINGLE_BROWSER, type.toUpperCase(), log));
    } else {
      this.writeCommonMsg(util.format(this.LOG_MULTI_BROWSER, browser, type.toUpperCase(), log));
    }
  };


  this.onSpecComplete = function(browser, result) {
    if (result.skipped) {
      this.specSkipped(browser, result);
    } else if (result.success) {
      this.specSuccess(browser, result);
    } else {
      this.specFailure(browser, result);
    }

    if (reportSlow && result.time > reportSlow) {
      var specName = result.suite.join(' ') + ' ' + result.description;
      var time = helper.formatTimeInterval(result.time);

      this.writeCommonMsg(util.format(this.SPEC_SLOW, browser, time, specName));
    }
  };


  this.specSuccess = this.specSkipped = function() {};


  this.specFailure = function(browser, result) {
    var specName = result.suite.join(' ') + ' ' + result.description;
    var msg = util.format(this.SPEC_FAILURE, browser, specName);

    result.log.forEach(function(log) {
      msg += formatError(log, '\t');
    });

    this.writeCommonMsg(msg);
  };


  this.onRunComplete = function(browsers, results) {
    if (browsers.length > 1 && !results.error && !results.disconnected) {
      if (!results.failed) {
        this.write(this.TOTAL_SUCCESS, results.success);
      } else {
        this.write(this.TOTAL_FAILED, results.failed, results.success);
      }
    }
  };

  this.USE_COLORS = false;

  this.LOG_SINGLE_BROWSER = 'LOG: %s\n';
  this.LOG_MULTI_BROWSER = '%s LOG: %s\n';

  this.SPEC_FAILURE = '%s %s FAILED' + '\n';
  this.SPEC_SLOW = '%s SLOW %s: %s\n';
  this.ERROR = '%s ERROR\n';

  this.FINISHED_ERROR = ' ERROR';
  this.FINISHED_SUCCESS = ' SUCCESS';
  this.FINISHED_DISCONNECTED = ' DISCONNECTED';

  this.X_FAILED = ' (%d FAILED)';

  this.TOTAL_SUCCESS = 'TOTAL: %d SUCCESS\n';
  this.TOTAL_FAILED = 'TOTAL: %d FAILED, %d SUCCESS\n';
};

BaseReporter.decoratorFactory = function(formatError, reportSlow) {
  return function(self) {
    BaseReporter.call(self, formatError, reportSlow);
  };
};

BaseReporter.decoratorFactory.$inject = ['formatError', 'config.reportSlowerThan'];


// PUBLISH
module.exports = BaseReporter;
