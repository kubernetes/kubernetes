(function() {
  var EscapeStore = require('./escape-store');
  var QuoteScanner = require('./quote-scanner');

  var Free = function Free() {
    this.matches = new EscapeStore('FREE_TEXT');
  };

  // Strip content tags by replacing them by the a special
  // marker for further restoring. It's done via string scanning
  // instead of regexps to speed up the process.
  Free.prototype.escape = function(data) {
    var self = this;

    return new QuoteScanner(data).each(function(match, store) {
      var placeholder = self.matches.store(match);
      store.push(placeholder);
    });
  };

  Free.prototype.restore = function(data) {
    return data.replace(this.matches.placeholderRegExp, this.matches.restore);
  };

  module.exports = Free;
})();
