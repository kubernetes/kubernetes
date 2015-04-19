var BaseReporter = require('./Base');


var DotsReporter = function(formatError, reportSlow) {
  BaseReporter.call(this, formatError, reportSlow);

  var DOTS_WRAP = 80;

  this.onRunStart = function(browsers) {
    this._browsers = browsers;
    this._dotsCount = 0;
  };

  this.writeCommonMsg = function(msg) {
    if (this._dotsCount) {
      this._dotsCount = 0;
      msg = '\n' + msg;
    }

    this.write(msg);

  };


  this.specSuccess = function() {
    this._dotsCount = (this._dotsCount + 1) % DOTS_WRAP;
    this.write(this._dotsCount ? '.' : '.\n');
  };

  this.onRunComplete = function(browsers, results) {
    this.writeCommonMsg(browsers.map(this.renderBrowser).join('\n') + '\n');

    if (browsers.length > 1 && !results.disconnected && !results.error) {
      if (!results.failed) {
        this.write(this.TOTAL_SUCCESS, results.success);
      } else {
        this.write(this.TOTAL_FAILED, results.failed, results.success);
      }
    }
  };
};


// PUBLISH
module.exports = DotsReporter;
