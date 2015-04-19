(function() {
  var QuoteScanner = function QuoteScanner(data) {
    this.data = data;
  };

  var findQuoteEnd = function(data, matched, cursor, oldCursor) {
    var commentStartMark = '/*';
    var commentEndMark = '*/';
    var escapeMark = '\\';
    var blockEndMark = '}';
    var dataPrefix = data.substring(oldCursor, cursor);
    var commentEndedAt = dataPrefix.lastIndexOf(commentEndMark, cursor);
    var commentStartedAt = dataPrefix.lastIndexOf(commentStartMark, cursor);
    var commentStarted = false;

    if (commentEndedAt >= cursor && commentStartedAt > -1)
      commentStarted = true;
    if (commentStartedAt < cursor && commentStartedAt > commentEndedAt)
      commentStarted = true;

    if (commentStarted) {
      var commentEndsAt = data.indexOf(commentEndMark, cursor);
      if (commentEndsAt > -1)
        return commentEndsAt;

      commentEndsAt = data.indexOf(blockEndMark, cursor);
      return commentEndsAt > -1 ? commentEndsAt - 1 : data.length;
    }

    while (true) {
      if (data[cursor] === undefined)
        break;
      if (data[cursor] == matched && (data[cursor - 1] != escapeMark || data[cursor - 2] == escapeMark))
        break;

      cursor++;
    }

    return cursor;
  };

  QuoteScanner.prototype.each = function(callback) {
    var data = this.data;
    var tempData = [];
    var nextStart = 0;
    var nextEnd = 0;
    var cursor = 0;
    var matchedMark = null;
    var singleMark = '\'';
    var doubleMark = '"';
    var dataLength = data.length;

    for (; nextEnd < data.length;) {
      var nextStartSingle = data.indexOf(singleMark, nextEnd + 1);
      var nextStartDouble = data.indexOf(doubleMark, nextEnd + 1);

      if (nextStartSingle == -1)
        nextStartSingle = dataLength;
      if (nextStartDouble == -1)
        nextStartDouble = dataLength;

      if (nextStartSingle < nextStartDouble) {
        nextStart = nextStartSingle;
        matchedMark = singleMark;
      } else {
        nextStart = nextStartDouble;
        matchedMark = doubleMark;
      }

      if (nextStart == -1)
        break;

      nextEnd = findQuoteEnd(data, matchedMark, nextStart + 1, cursor);
      if (nextEnd == -1)
        break;

      var text = data.substring(nextStart, nextEnd + 1);
      tempData.push(data.substring(cursor, nextStart));
      if (text.length > 0)
        callback(text, tempData, nextStart);

      cursor = nextEnd + 1;
    }

    return tempData.length > 0 ?
      tempData.join('') + data.substring(cursor, data.length) :
      data;
  };

  module.exports = QuoteScanner;
})();
