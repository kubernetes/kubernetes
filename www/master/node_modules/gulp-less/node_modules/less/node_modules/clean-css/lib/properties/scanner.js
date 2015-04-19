(function() {
  var OPEN_BRACE = '{';
  var SEMICOLON = ';';
  var COLON = ':';

  var PropertyScanner = function PropertyScanner(data) {
    this.data = data;
  };

  PropertyScanner.prototype.nextAt = function(cursor) {
    var lastColon = this.data.lastIndexOf(COLON, cursor);
    var lastOpenBrace = this.data.lastIndexOf(OPEN_BRACE, cursor);
    var lastSemicolon = this.data.lastIndexOf(SEMICOLON, cursor);
    var startAt = Math.max(lastOpenBrace, lastSemicolon);

    return this.data.substring(startAt + 1, lastColon).trim();
  };

  module.exports = PropertyScanner;
})();
