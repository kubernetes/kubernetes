var EscapeStore = require('./escape-store');
var QuoteScanner = require('../utils/quote-scanner');

var SPECIAL_COMMENT_PREFIX = '/*!';
var COMMENT_PREFIX = '/*';
var COMMENT_SUFFIX = '*/';

var lineBreak = require('os').EOL;

function CommentsProcessor(context, keepSpecialComments, keepBreaks, saveWaypoints) {
  this.comments = new EscapeStore('COMMENT');
  this.specialComments = new EscapeStore('COMMENT_SPECIAL');

  this.context = context;
  this.keepAll = keepSpecialComments == '*';
  this.keepOne = keepSpecialComments == '1' || keepSpecialComments === 1;
  this.keepBreaks = keepBreaks;
  this.saveWaypoints = saveWaypoints;
}

function quoteScannerFor(data) {
  var quoteMap = [];
  new QuoteScanner(data).each(function (quotedString, _, startsAt) {
    quoteMap.push([startsAt, startsAt + quotedString.length]);
  });

  return function (position) {
    for (var i = 0, l = quoteMap.length; i < l; i++) {
      if (quoteMap[i][0] < position && quoteMap[i][1] > position)
        return true;
    }

    return false;
  };
}

CommentsProcessor.prototype.escape = function (data) {
  var tempData = [];
  var nextStart = 0;
  var nextEnd = 0;
  var cursor = 0;
  var indent = 0;
  var breaksCount;
  var lastBreakAt;
  var newIndent;
  var isQuotedAt = quoteScannerFor(data);
  var saveWaypoints = this.saveWaypoints;

  for (; nextEnd < data.length;) {
    nextStart = data.indexOf(COMMENT_PREFIX, cursor);
    if (nextStart == -1)
      break;

    if (isQuotedAt(nextStart)) {
      tempData.push(data.substring(cursor, nextStart + COMMENT_PREFIX.length));
      cursor = nextStart + COMMENT_PREFIX.length;
      continue;
    }

    nextEnd = data.indexOf(COMMENT_SUFFIX, nextStart + COMMENT_PREFIX.length);
    if (nextEnd == -1) {
      this.context.warnings.push('Broken comment: \'' + data.substring(nextStart) + '\'.');
      nextEnd = data.length - 2;
    }

    tempData.push(data.substring(cursor, nextStart));

    var comment = data.substring(nextStart, nextEnd + COMMENT_SUFFIX.length);
    var isSpecialComment = comment.indexOf(SPECIAL_COMMENT_PREFIX) === 0;

    if (saveWaypoints) {
      breaksCount = comment.split(lineBreak).length - 1;
      lastBreakAt = comment.lastIndexOf(lineBreak);
      newIndent = lastBreakAt > 0 ?
        comment.substring(lastBreakAt + lineBreak.length).length :
        indent + comment.length;
    }

    if (saveWaypoints || isSpecialComment) {
      var metadata = saveWaypoints ? [breaksCount, newIndent] : null;
      var placeholder = isSpecialComment ?
        this.specialComments.store(comment, metadata) :
        this.comments.store(comment, metadata);
      tempData.push(placeholder);
    }

    if (saveWaypoints)
      indent = newIndent + 1;
    cursor = nextEnd + COMMENT_SUFFIX.length;
  }

  return tempData.length > 0 ?
    tempData.join('') + data.substring(cursor, data.length) :
    data;
};

function restore(context, data, from, isSpecial) {
  var tempData = [];
  var restored = 0;
  var cursor = 0;
  var addBreak;

  for (; cursor < data.length;) {
    var nextMatch = from.nextMatch(data, cursor);
    if (nextMatch.start < 0)
      break;

    tempData.push(data.substring(cursor, nextMatch.start));
    var comment = from.restore(nextMatch.match);

    if (isSpecial && (context.keepAll || (context.keepOne && restored === 0))) {
      restored++;
      addBreak = context.keepBreaks && data[nextMatch.end] != '\n' && data.lastIndexOf('\r\n', nextMatch.end + 1) != nextMatch.end;
      tempData.push(comment, addBreak ? lineBreak : '');
    } else {
      nextMatch.end += context.keepBreaks ? lineBreak.length : 0;
    }

    cursor = nextMatch.end;
  }

  return tempData.length > 0 ?
    tempData.join('') + data.substring(cursor, data.length) :
    data;
}

CommentsProcessor.prototype.restore = function (data) {
  data = restore(this, data, this.comments, false);
  data = restore(this, data, this.specialComments, true);
  return data;
};

module.exports = CommentsProcessor;
