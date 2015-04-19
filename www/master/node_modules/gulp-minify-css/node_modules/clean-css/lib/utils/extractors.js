var Splitter = require('./splitter');
var SourceMaps = require('../utils/source-maps');

var Extractors = {
  properties: function (string, context) {
    var tokenized = [];
    var list = [];
    var buffer = [];
    var all = [];
    var property;
    var isPropertyEnd;
    var isWhitespace;
    var wasWhitespace;
    var isSpecial;
    var wasSpecial;
    var current;
    var last;
    var secondToLast;
    var wasCloseParenthesis;
    var isEscape;
    var token;
    var addSourceMap = context.addSourceMap;

    for (var i = 0, l = string.length; i < l; i++) {
      current = string[i];
      isPropertyEnd = current === ';';

      isEscape = !isPropertyEnd && current == '_' && string.indexOf('__ESCAPED_COMMENT', i) === i;
      if (isEscape) {
        if (buffer.length > 0) {
          i--;
          isPropertyEnd = true;
        } else {
          var endOfEscape = string.indexOf('__', i + 1) + 2;
          var comment = string.substring(i, endOfEscape);
          i = endOfEscape - 1;

          if (comment.indexOf('__ESCAPED_COMMENT_SPECIAL') === -1) {
            if (addSourceMap)
              SourceMaps.track(comment, context, true);
            continue;
          }
          else {
            buffer = all = [comment];
          }
        }
      }

      if (isPropertyEnd || isEscape) {
        if (wasWhitespace && buffer[buffer.length - 1] === ' ')
          buffer.pop();
        if (buffer.length > 0) {
          property = buffer.join('');
          token = { value: property };
          tokenized.push(token);
          list.push(property);

          if (addSourceMap)
            token.metadata = SourceMaps.saveAndTrack(all.join(''), context, !isEscape);
        }
        buffer = [];
        all = [];
      } else {
        isWhitespace = current === ' ' || current === '\t' || current === '\n';
        isSpecial = current === ':' || current === '[' || current === ']' || current === ',' || current === '(' || current === ')';

        if (wasWhitespace && isSpecial) {
          last = buffer[buffer.length - 1];
          secondToLast = buffer[buffer.length - 2];
          if (secondToLast != '+' && secondToLast != '-' && secondToLast != '/' && secondToLast != '*' && last != '(')
            buffer.pop();
          buffer.push(current);
        } else if (isWhitespace && wasSpecial && !wasCloseParenthesis) {
        } else if (isWhitespace && !wasWhitespace && buffer.length > 0) {
          buffer.push(' ');
        } else if (isWhitespace && buffer.length === 0) {
        } else if (isWhitespace && wasWhitespace) {
        } else {
          buffer.push(isWhitespace ? ' ' : current);
        }

        all.push(current);
      }

      wasSpecial = isSpecial;
      wasWhitespace = isWhitespace;
      wasCloseParenthesis = current === ')';
    }

    if (wasWhitespace && buffer[buffer.length - 1] === ' ')
      buffer.pop();
    if (buffer.length > 0) {
      property = buffer.join('');
      token = { value: property };
      tokenized.push(token);
      list.push(property);

      if (addSourceMap)
        token.metadata = SourceMaps.saveAndTrack(all.join(''), context, false);
    } else if (all.indexOf('\n') > -1) {
      SourceMaps.track(all.join(''), context);
    }

    return {
      list: list,
      tokenized: tokenized
    };
  },

  selectors: function (string, context) {
    var tokenized = [];
    var list = [];
    var selectors = new Splitter(',').split(string);
    var addSourceMap = context.addSourceMap;

    for (var i = 0, l = selectors.length; i < l; i++) {
      var selector = selectors[i];

      list.push(selector);

      var token = { value: selector };
      tokenized.push(token);

      if (addSourceMap)
        token.metadata = SourceMaps.saveAndTrack(selector, context, true);
    }

    return {
      list: list,
      tokenized: tokenized
    };
  }
};

module.exports = Extractors;
