var EscapeStore = require('./escape-store');
var QuoteScanner = require('../utils/quote-scanner');

var lineBreak = require('os').EOL;

function FreeTextProcessor(saveWaypoints) {
  this.matches = new EscapeStore('FREE_TEXT');
  this.saveWaypoints = saveWaypoints;
}

// Strip content tags by replacing them by the a special
// marker for further restoring. It's done via string scanning
// instead of regexps to speed up the process.
FreeTextProcessor.prototype.escape = function(data) {
  var self = this;
  var breaksCount;
  var lastBreakAt;
  var indent;
  var metadata;
  var saveWaypoints = this.saveWaypoints;

  return new QuoteScanner(data).each(function(match, store) {
    if (saveWaypoints) {
      breaksCount = match.split(lineBreak).length - 1;
      lastBreakAt = match.lastIndexOf(lineBreak);
      indent = lastBreakAt > 0 ?
        match.substring(lastBreakAt + lineBreak.length).length :
        match.length;
      metadata = [breaksCount, indent];
    }

    var placeholder = self.matches.store(match, metadata);
    store.push(placeholder);
  });
};

function normalize(text, data, cursor) {
  // FIXME: this is a hack
  var lastSemicolon = data.lastIndexOf(';', cursor);
  var lastOpenBrace = data.lastIndexOf('{', cursor);
  var lastOne = 0;

  if (lastSemicolon > -1 && lastOpenBrace > -1)
    lastOne = Math.max(lastSemicolon, lastOpenBrace);
  else if (lastSemicolon == -1)
    lastOne = lastOpenBrace;
  else
    lastOne = lastSemicolon;

  var context = data.substring(lastOne + 1, cursor);

  if (/\[[\w\d\-]+[\*\|\~\^\$]?=$/.test(context))
    text = text.replace(/\\\n|\\\r\n/g, '');

  if (/^['"][a-zA-Z][a-zA-Z\d\-_]+['"]$/.test(text) && !/format\($/.test(context)) {
    var isFont = /^(font|font\-family):/.test(context);
    var isAttribute = /\[[\w\d\-]+[\*\|\~\^\$]?=$/.test(context);
    var isKeyframe = /@(-moz-|-o-|-webkit-)?keyframes /.test(context);
    var isAnimation = /^(-moz-|-o-|-webkit-)?animation(-name)?:/.test(context);

    if (isFont || isAttribute || isKeyframe || isAnimation)
      text = text.substring(1, text.length - 1);
  }

  return text;
}

FreeTextProcessor.prototype.restore = function(data) {
  var tempData = [];
  var cursor = 0;

  for (; cursor < data.length;) {
    var nextMatch = this.matches.nextMatch(data, cursor);
    if (nextMatch.start < 0)
      break;

    tempData.push(data.substring(cursor, nextMatch.start));
    var text = normalize(this.matches.restore(nextMatch.match), data, nextMatch.start);
    tempData.push(text);

    cursor = nextMatch.end;
  }

  return tempData.length > 0 ?
    tempData.join('') + data.substring(cursor, data.length) :
    data;
};

module.exports = FreeTextProcessor;
