var EscapeStore = require('./escape-store');

module.exports = function Expressions() {
  var expressions = new EscapeStore('EXPRESSION');

  var findEnd = function(data, start) {
    var end = start + 'expression'.length;
    var level = 0;
    var quoted = false;

    while (true) {
      var next = data[end++];

      if (quoted) {
        quoted = next != '\'' && next != '"';
      } else {
        quoted = next == '\'' || next == '"';

        if (next == '(')
          level++;
        if (next == ')')
          level--;
        if (next == '}' && level == 1) {
          end--;
          level--;
        }
      }

      if (level === 0 && next == ')')
        break;
      if (!next) {
        end = data.substring(0, end).lastIndexOf('}');
        break;
      }
    }

    return end;
  };

  return {
    // Escapes expressions by replacing them by a special
    // marker for further restoring. It's done via string scanning
    // instead of regexps to speed up the process.
    escape: function(data) {
      var nextStart = 0;
      var nextEnd = 0;
      var cursor = 0;
      var tempData = [];

      for (; nextEnd < data.length;) {
        nextStart = data.indexOf('expression(', nextEnd);
        if (nextStart == -1)
          break;

        nextEnd = findEnd(data, nextStart);

        var expression = data.substring(nextStart, nextEnd);
        var placeholder = expressions.store(expression);
        tempData.push(data.substring(cursor, nextStart));
        tempData.push(placeholder);
        cursor = nextEnd;
      }

      return tempData.length > 0 ?
        tempData.join('') + data.substring(cursor, data.length) :
        data;
    },

    restore: function(data) {
      return data.replace(expressions.placeholderRegExp, expressions.restore);
    }
  };
};
