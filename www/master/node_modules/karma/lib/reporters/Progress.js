var BaseReporter = require('./Base');


var ProgressReporter = function(formatError, reportSlow) {
  BaseReporter.call(this, formatError, reportSlow);


  this.writeCommonMsg = function(msg) {
    this.write(this._remove() + msg + this._render());
  };


  this.specSuccess = function() {
    this.write(this._refresh());
  };


  this.onBrowserComplete = function() {
    this.write(this._refresh());
  };


  this.onRunStart = function(browsers) {
    this._browsers = browsers;
    this._isRendered = false;
  };


  this._remove = function() {
    if (!this._isRendered) {
      return '';
    }

    var cmd = '';
    this._browsers.forEach(function() {
      cmd += '\x1B[1A' + '\x1B[2K';
    });

    this._isRendered = false;

    return cmd;
  };

  this._render = function() {
    this._isRendered = true;

    return this._browsers.map(this.renderBrowser).join('\n') + '\n';
  };

  this._refresh = function() {
    return this._remove() + this._render();
  };
};


// PUBLISH
module.exports = ProgressReporter;
