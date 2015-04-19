(function() {
  var QuoteScanner = require('./quote-scanner');
  var PropertyScanner = require('../properties/scanner');

  var NameQuotes = function NameQuotes() {};

  var STRIPPABLE = /^['"][a-zA-Z][a-zA-Z\d\-_]+['"]$/;

  var properties = [
    'animation',
    '-moz-animation',
    '-o-animation',
    '-webkit-animation',
    'animation-name',
    '-moz-animation-name',
    '-o-animation-name',
    '-webkit-animation-name',
    'font',
    'font-family'
  ];

  NameQuotes.prototype.process = function(data) {
    var scanner = new PropertyScanner(data);

    return new QuoteScanner(data).each(function(match, store, cursor) {
      var lastProperty = scanner.nextAt(cursor);
      if (properties.indexOf(lastProperty) > -1) {
        if (STRIPPABLE.test(match))
          match = match.substring(1, match.length - 1);
      }

      store.push(match);
    });
  };

  module.exports = NameQuotes;
})();
