var EscapeStore = require('./escape-store');

var EXPRESSION_NAME = 'expression';
var EXPRESSION_START = '(';
var EXPRESSION_END = ')';
var EXPRESSION_PREFIX = EXPRESSION_NAME + EXPRESSION_START;
var BODY_START = '{';
var BODY_END = '}';

var lineBreak = require('os').EOL;

function findEnd(data, start) {
  var end = start + EXPRESSION_NAME.length;
  var level = 0;
  var quoted = false;
  var braced = false;

  while (true) {
    var current = data[end++];

    if (quoted) {
      quoted = current != '\'' && current != '"';
    } else {
      quoted = current == '\'' || current == '"';

      if (current == EXPRESSION_START)
        level++;
      if (current == EXPRESSION_END)
        level--;
      if (current == BODY_START)
        braced = true;
      if (current == BODY_END && !braced && level == 1) {
        end--;
        level--;
      }
    }

    if (level === 0 && current == EXPRESSION_END)
      break;
    if (!current) {
      end = data.substring(0, end).lastIndexOf(BODY_END);
      break;
    }
  }

  return end;
}

function ExpressionsProcessor(saveWaypoints) {
  this.expressions = new EscapeStore('EXPRESSION');
  this.saveWaypoints = saveWaypoints;
}

ExpressionsProcessor.prototype.escape = function (data) {
  var nextStart = 0;
  var nextEnd = 0;
  var cursor = 0;
  var tempData = [];
  var indent = 0;
  var breaksCount;
  var lastBreakAt;
  var newIndent;
  var saveWaypoints = this.saveWaypoints;

  for (; nextEnd < data.length;) {
    nextStart = data.indexOf(EXPRESSION_PREFIX, nextEnd);
    if (nextStart == -1)
      break;

    nextEnd = findEnd(data, nextStart);

    var expression = data.substring(nextStart, nextEnd);
    if (saveWaypoints) {
      breaksCount = expression.split(lineBreak).length - 1;
      lastBreakAt = expression.lastIndexOf(lineBreak);
      newIndent = lastBreakAt > 0 ?
        expression.substring(lastBreakAt + lineBreak.length).length :
        indent + expression.length;
    }

    var metadata = saveWaypoints ? [breaksCount, newIndent] : null;
    var placeholder = this.expressions.store(expression, metadata);
    tempData.push(data.substring(cursor, nextStart));
    tempData.push(placeholder);

    if (saveWaypoints)
      indent = newIndent + 1;
    cursor = nextEnd;
  }

  return tempData.length > 0 ?
    tempData.join('') + data.substring(cursor, data.length) :
    data;
};

ExpressionsProcessor.prototype.restore = function (data) {
  var tempData = [];
  var cursor = 0;

  for (; cursor < data.length;) {
    var nextMatch = this.expressions.nextMatch(data, cursor);
    if (nextMatch.start < 0)
      break;

    tempData.push(data.substring(cursor, nextMatch.start));
    var comment = this.expressions.restore(nextMatch.match);
    tempData.push(comment);

    cursor = nextMatch.end;
  }

  return tempData.length > 0 ?
    tempData.join('') + data.substring(cursor, data.length) :
    data;
};

module.exports = ExpressionsProcessor;
